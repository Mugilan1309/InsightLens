import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "text_data" / "tweets.json"
MODELS_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = MODELS_DIR / "insightlens_text.h5"
TOKENIZER_SAVE_PATH = MODELS_DIR / "tokenizer.pickle"

# Hyperparameters
VOCAB_SIZE = 10000   # Max words to keep
MAX_LENGTH = 50      # Max words per tweet
EMBEDDING_DIM = 64
EPOCHS = 5
BATCH_SIZE = 64

def validate_paths():
    if not DATA_PATH.exists():
        sys.exit(f"❌ Error: Text data not found at {DATA_PATH}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_process_data():
    """
    Loads JSON, bins 'likes' into 3 balanced classes using Rank.
    """
    print("[Data] Loading Twitter dataset...")
    try:
        df = pd.read_json(DATA_PATH, lines=True)
    except ValueError:
        df = pd.read_json(DATA_PATH)

    # 1. Clean Data
    df = df[['tweet', 'likes']].dropna()
    
    # 2. Binning with Rank (The Fix)
    # This handles the "Too many zeros" issue by forcing a 33/33/33 split
    # regardless of duplicate values.
    df['likes_rank'] = df['likes'].rank(method='first')
    df['label'] = pd.qcut(df['likes_rank'], q=3, labels=[0, 1, 2])
    
    print(f"[Data] Loaded {len(df)} rows.")
    print(f"[Data] Class distribution (Balanced):\n{df['label'].value_counts()}")

    return df

def prepare_text_features(df):
    """
    Converts text to padded sequences and saves the tokenizer.
    """
    print("[Preprocessing] Fitting Tokenizer...")
    
    # SAFETY FIX: Ensure all inputs are strings (fixes rare crashing bug)
    df['tweet'] = df['tweet'].astype(str)
    
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['tweet'])

    # Convert text to numbers
    sequences = tokenizer.texts_to_sequences(df['tweet'])
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

    # Save Tokenizer
    with open(TOKENIZER_SAVE_PATH, 'wb') as handle:
        # FIX: Corrected protocol usage
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Artifact] Tokenizer saved to {TOKENIZER_SAVE_PATH}")

    # Prepare Labels
    labels = to_categorical(df['label'], num_classes=3)
    
    return padded, labels

def build_model():
    """
    Builds a Bi-Directional LSTM for text classification.
    """
    print("[Model] Building NLP Architecture...")
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        # Bidirectional LSTM lets the model understand context from both left and right
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        # 3 Output neurons for Low/Mid/High
        Dense(3, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def main():
    print("--- 📝 InsightLens Text Module Training 📝 ---")
    validate_paths()

    # 1. Data
    df = load_and_process_data()
    X, y = prepare_text_features(df)

    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model
    model = build_model()
    model.summary()

    # 4. Train
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE
    )

    # 5. Save
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()