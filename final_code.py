# %% [markdown]
# # Image Captioning con TensorFlow y Flickr8k
# 
# ## Sistema de descripción automática de imágenes
# 
# **Arquitectura**: Encoder-Decoder con VGG16 + LSTM con atención

# %% [markdown]
# ## 1. Configuración inicial

# %%
# Verificar e instalar dependencias
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__} ya está instalado")
except ImportError:
    #pip install tensorflow numpy matplotlib pillow nltk tqdm pycocotools
    import tensorflow as tf

# %% [markdown]
# ## 2. Imports y configuración

# %%
# Importar librerías
import os
import numpy as np
from pathlib import Path
from PIL import Image
import string
import datetime
import re
from collections import Counter

# TensorFlow/Keras
from tensorflow.keras.layers import (Input, Dense, Embedding, LSTM, Concatenate, 
                                   Dropout, BatchNormalization, Layer, 
                                   GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                      ReduceLROnPlateau, TensorBoard)

# NLTK y evaluación
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Visualización y progreso
from tqdm import tqdm
from IPython.display import display, Markdown

# Configurar paths
class Config:
    def __init__(self):
        self.BASE_DIR = Path.cwd()
        self.IMAGE_DIR = self.BASE_DIR / "Flicker8k_Dataset"
        self.TRAIN_FILE = self.BASE_DIR / "Flickr_8k.trainImages.txt"
        self.TEST_FILE = self.BASE_DIR / "Flickr_8k.testImages.txt"
        self.DESCRIPTIONS_FILE = self.BASE_DIR / "Flickr8k.token.txt"
        
        # Crear directorios si no existen
        self.IMAGE_DIR.mkdir(parents=True, exist_ok=True)

config = Config()

# Descargar recursos NLTK
nltk.download('punkt', quiet=True)

# %% [markdown]
# ## 3. Carga y preparación de datos

# %%
def load_image_ids(file_path):
    """Carga IDs de imágenes desde archivo"""
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip().split('.')[0] for line in f if line.strip()]

def load_descriptions(file_path):
    """Carga y normaliza descripciones"""
    descriptions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
                
            img_id, desc = line.split('\t')
            img_id = img_id.split('.')[0]
            desc = desc.lower().translate(str.maketrans('', '', string.punctuation))
            
            descriptions.setdefault(img_id, []).append(desc)
    
    return descriptions

# Cargar datos
train_ids = load_image_ids(config.TRAIN_FILE)
test_ids = load_image_ids(config.TEST_FILE)
all_descriptions = load_descriptions(config.DESCRIPTIONS_FILE)

# Filtrar descripciones
train_descriptions = {k: all_descriptions[k] for k in train_ids if k in all_descriptions}
test_descriptions = {k: all_descriptions[k] for k in test_ids if k in all_descriptions}

print(f"✔ Imágenes de entrenamiento: {len(train_ids)}")
print(f"✔ Imágenes de prueba: {len(test_ids)}")
print(f"✔ Descripciones de entrenamiento: {len(train_descriptions)}")
print(f"✔ Descripciones de prueba: {len(test_descriptions)}")

# %% [markdown]
# ## 4. Construcción del vocabulario

# %%
def build_vocabulary(descriptions, min_count=5):
    """Construye vocabulario con filtrado mejorado"""
    word_counts = Counter()
    
    for desc_list in tqdm(descriptions.values(), desc="Procesando descripciones"):
        for desc in desc_list:
            tokens = word_tokenize(desc.lower())
            tokens = [t for t in tokens if t.isalpha()]  # Solo palabras alfabéticas
            word_counts.update(tokens)
    
    vocab = ['<start>', '<end>', '<pad>', '<unk>'] + \
            [word for word, count in word_counts.items() if count >= min_count]
    
    print(f"\nVocabulario creado: {len(vocab)} palabras")
    print("20 palabras más frecuentes:", word_counts.most_common(20))
    
    return vocab, {word: idx for idx, word in enumerate(vocab)}, \
           {idx: word for idx, word in enumerate(vocab)}

vocabulary, word_to_idx, idx_to_word = build_vocabulary(train_descriptions, min_count=3)
vocab_size = len(vocabulary)

# %% [markdown]
# ## 5. Preprocesamiento de descripciones

# %%
def preprocess_descriptions(descriptions, word_to_idx):
    """Convierte texto a secuencias de índices"""
    processed = {}
    max_length = 0
    
    for img_id, desc_list in tqdm(descriptions.items(), desc="Preprocesando"):
        processed[img_id] = []
        for desc in desc_list:
            words = word_tokenize(desc.lower())
            seq = [word_to_idx.get(word, word_to_idx['<unk>']) for word in words 
                  if word in word_to_idx or word.isalpha()]
            if len(seq) > max_length:
                max_length = len(seq)
            processed[img_id].append(seq)
    
    return processed, max_length

train_descriptions_processed, max_seq_length = preprocess_descriptions(train_descriptions, word_to_idx)
test_descriptions_processed, _ = preprocess_descriptions(test_descriptions, word_to_idx)

print(f"\nLongitud máxima de secuencia: {max_seq_length}")

# %% [markdown]
# ## 6. Modelo Encoder-Decoder con Atención

# %%
class BahdanauAttention(Layer):
    """Mecanismo de atención mejorado"""
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

def create_encoder():
    """Encoder con VGG16 preentrenado"""
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    vgg.trainable = False
    
    inputs = Input(shape=(224, 224, 3))
    features = vgg(inputs)
    features = GlobalAveragePooling2D()(features)
    
    return Model(inputs, features, name='encoder')

def create_decoder(vocab_size, max_length, embedding_dim=256, lstm_units=512):
    """Decoder con LSTM y atención"""
    # Inputs
    image_input = Input(shape=(512,), name='image_features')
    caption_input = Input(shape=(max_length,), name='caption_input')
    
    # Capa de imagen
    img_features = Dense(embedding_dim, activation='relu')(image_input)
    img_features = Dropout(0.5)(img_features)
    
    # Embedding de texto
    text_embed = Embedding(vocab_size, embedding_dim, mask_zero=True)(caption_input)
    text_embed = Dropout(0.3)(text_embed)
    
    # LSTM con atención
    lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    lstm_output, state_h, state_c = lstm(text_embed)
    
    # Atención
    attention = BahdanauAttention(lstm_units)
    context_vector, _ = attention(lstm_output, img_features)
    
    # Decodificación
    combined = Concatenate()([context_vector, img_features])
    decoder_output = Dense(lstm_units, activation='relu')(combined)
    output = Dense(vocab_size, activation='softmax')(decoder_output)
    
    # Modelo
    decoder = Model(inputs=[image_input, caption_input], outputs=output)
    
    # Compilación
    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
    decoder.compile(optimizer=optimizer, 
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'])
    
    return decoder

# Crear modelos
encoder = create_encoder()
decoder = create_decoder(vocab_size, max_seq_length)

# Resumen
decoder.summary()

# %% [markdown]
# ## 7. Extracción de características

# %%
def preprocess_image(image_path):
    """Preprocesa imagen para VGG16"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return preprocess_input(img_array)

def extract_features(image_ids, image_dir):
    """Extrae características usando el encoder"""
    features = {}
    for img_id in tqdm(image_ids, desc="Extrayendo características"):
        img_path = image_dir / f"{img_id}.jpg"
        img_array = np.expand_dims(preprocess_image(img_path), axis=0)
        features[img_id] = encoder.predict(img_array, verbose=0)[0]
    return features

# Extraer características (esto puede tardar)
train_features = extract_features(train_ids, config.IMAGE_DIR)
test_features = extract_features(test_ids, config.IMAGE_DIR)

# %% [markdown]
# ## 8. Generador de datos

# %%
def create_tf_dataset(descriptions, features, batch_size=64):
    """Crea dataset de TensorFlow optimizado"""
    def gen():
        for img_id, desc_list in descriptions.items():
            feature = features[img_id]
            for desc in desc_list:
                for i in range(1, len(desc)):
                    yield {
                        'image_features': feature,
                        'caption_input': pad_sequences([desc[:i]], maxlen=max_seq_length, padding='post')[0]
                    }, desc[i]
    
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                'image_features': tf.TensorSpec(shape=(512,), dtype=tf.float32),
                'caption_input': tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32)
            },
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Crear datasets
train_dataset = create_tf_dataset(train_descriptions_processed, train_features)
test_dataset = create_tf_dataset(test_descriptions_processed, test_features)

# Dividir train/val
val_size = int(0.2 * len(train_descriptions_processed))
val_dataset = train_dataset.take(val_size)
train_dataset = train_dataset.skip(val_size)

# %% [markdown]
# ## 9. Entrenamiento del modelo

# %%
# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        min_delta=0.01
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    TensorBoard(
        log_dir='logs',
        histogram_freq=1
    )
]

# Entrenamiento
history = decoder.fit(
    train_dataset,
    epochs=30,
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)

# %% [markdown]
# ## 10. Generación de descripciones

# %%
def generate_caption(model, encoder, image_path, word_to_idx, idx_to_word, max_length=36, temperature=0.7):
    """Genera descripción para una imagen"""
    # Preprocesar imagen
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extraer características
    feature = encoder.predict(img_array, verbose=0).reshape(1, -1)
    
    # Inicializar secuencia
    start_token = word_to_idx['<start>']
    end_token = word_to_idx['<end>']
    sequence = [start_token]
    
    for _ in range(max_length):
        padded_seq = pad_sequences([sequence], maxlen=max_length, padding='post')
        preds = model.predict(
            {'image_features': feature, 'caption_input': padded_seq},
            verbose=0
        )[0]
        
        # Aplicar temperatura
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        # Evitar tokens especiales
        valid_indices = [i for i in range(len(preds)) 
                        if idx_to_word[i] not in ['<pad>', '<unk>']]
        valid_probs = preds[valid_indices]
        valid_probs /= np.sum(valid_probs)
        
        next_token = np.random.choice(valid_indices, p=valid_probs)
        sequence.append(next_token)
        
        if next_token == end_token:
            break
    
    # Convertir a texto
    caption = ' '.join([idx_to_word[idx] for idx in sequence 
                       if idx_to_word[idx] not in ['<start>', '<end>', '<pad>']])
    
    # Post-procesamiento
    caption = caption.capitalize()
    if caption and caption[-1] not in {'.', '!', '?'}:
        caption += '.'
        
    return caption

# Probar con una imagen de prueba
sample_img = list(test_descriptions.keys())[0]
sample_path = config.IMAGE_DIR / f"{sample_img}.jpg"

display(Image.open(sample_path))
print("\nDescripciones reales:")
print("\n".join(test_descriptions[sample_img]))
print("\nDescripción generada:")
print(generate_caption(decoder, encoder, sample_path, word_to_idx, idx_to_word))

# %% [markdown]
# ## 11. Evaluación del modelo

# %%
def evaluate_model(model, encoder, test_data, features, word_to_idx, idx_to_word, num_samples=50):
    """Evalúa el modelo con métricas BLEU"""
    actual = []
    predicted = []
    
    for img_id, desc_list in tqdm(list(test_data.items())[:num_samples], desc="Evaluando"):
        img_path = config.IMAGE_DIR / f"{img_id}.jpg"
        
        # Generar descripción
        gen_desc = generate_caption(model, encoder, img_path, word_to_idx, idx_to_word)
        
        # Procesar descripciones reales
        real_descs = [word_tokenize(desc.lower()) for desc in desc_list]
        
        actual.append(real_descs)
        predicted.append(word_tokenize(gen_desc.lower()))
    
    # Calcular BLEU
    smooth = SmoothingFunction().method4
    bleu1 = corpus_bleu(actual, predicted, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    print(f"\nBLEU-1: {bleu1:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    
    return bleu1, bleu4

# Evaluar
bleu1, bleu4 = evaluate_model(decoder, encoder, test_descriptions, test_features, word_to_idx, idx_to_word)