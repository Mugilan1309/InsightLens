import sys
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
# Define paths using pathlib relative to this script's location or CWD
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "koniq"
IMG_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "koniq10k_scores.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = MODELS_DIR / "insightlens_vision.h5"

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def validate_paths() -> None:
    """Ensures critical directories and files exist before starting."""
    if not IMG_DIR.exists():
        sys.exit(f"❌ Error: Image directory not found at {IMG_DIR}")
    if not CSV_PATH.exists():
        sys.exit(f"❌ Error: CSV file not found at {CSV_PATH}")
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean_data() -> pd.DataFrame:
    """
    Loads metadata and filters out entries where the image file is missing.
    Returns:
        pd.DataFrame: Cleaned dataframe ready for the generator.
    """
    print(f"Loading data from: {CSV_PATH.name}...")
    df = pd.read_csv(CSV_PATH)
    initial_count = len(df)
    
    # Efficiently check for file existence
    # We construct the full path for every image and check .is_file()
    df['file_path'] = df['image_name'].apply(lambda x: IMG_DIR / x)
    df['exists'] = df['file_path'].apply(lambda p: p.is_file())
    
    df_clean = df[df['exists']].copy()
    
    # Drop helper columns to keep it clean
    df_clean = df_clean.drop(columns=['file_path', 'exists'])
    
    print(f"✅ Data Loaded. Rows: {initial_count} -> Valid Images: {len(df_clean)}")
    return df_clean

def create_generators(df: pd.DataFrame):
    """
    Creates Keras ImageDataGenerators for training and validation.
    """
    # Normalize pixel values (0-255 -> 0-1)
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Note: flow_from_dataframe requires path as string, not Path object, in older TF versions.
    # We pass the directory as a string to be safe.
    print("Initializing Data Generators...")
    train_gen = datagen.flow_from_dataframe(
        dataframe=df,
        directory=str(IMG_DIR),
        x_col="image_name",
        y_col="MOS", # Mean Opinion Score
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="raw", # Regression task
        subset="training",
        shuffle=True
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=df,
        directory=str(IMG_DIR),
        x_col="image_name",
        y_col="MOS",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="raw",
        subset="validation",
        shuffle=False
    )
    
    return train_gen, val_gen

def build_model() -> Model:
    """
    Constructs the MobileNetV2 Transfer Learning architecture.
    """
    print("Building MobileNetV2 architecture...")
    
    # Load pre-trained MobileNetV2 (without the top classification layer)
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze the base model to prevent destroying learned features
    base_model.trainable = False

    # Add custom regression head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x) # Regularization
    predictions = Dense(1, activation='linear')(x) # Single output for regression score

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE), 
        loss='mse', 
        metrics=['mae']
    )
    return model

def main():
    print("--- 👁️ InsightLens Vision Module Training 👁️ ---")
    
    # 1. Validation
    validate_paths()
    
    # 2. Data Preparation
    df = load_and_clean_data()
    train_gen, val_gen = create_generators(df)
    
    # 3. Model Architecture
    model = build_model()
    model.summary()
    
    # 4. Training Loop
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    try:
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            steps_per_epoch=len(train_gen),
            validation_steps=len(val_gen)
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user. Saving current state...")

    # 5. Save Artifact
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model saved successfully.")

if __name__ == "__main__":
    main()