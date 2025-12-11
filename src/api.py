import uvicorn
import numpy as np
import tensorflow as tf
import pickle
import io
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# App Setup
app = FastAPI(title="InsightLens API")

# Enable CORS (allows your JS to talk to this Python backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Variables for Models
vision_model = None
text_model = None
tokenizer = None

# Constants
IMG_SIZE = (224, 224)
MAX_LENGTH = 50

# --- LIFECYCLE EVENTS ---
@app.on_event("startup")
def load_models():
    global vision_model, text_model, tokenizer
    print("⚡ Loading InsightLens Models...")
    
    # Load Vision
    vision_model = tf.keras.models.load_model(MODELS_DIR / "insightlens_vision.h5")
    
    # Load Text
    text_model = tf.keras.models.load_model(MODELS_DIR / "insightlens_text.h5")
    
    # Load Tokenizer
    with open(MODELS_DIR / "tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    print("✅ Models Loaded & Ready!")

# --- ENDPOINTS ---

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    # 1. Read Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # 2. Preprocess
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # 3. Predict
    prediction = vision_model.predict(img_array, verbose=0)
    raw_score = float(prediction[0][0])
    
    score = min(max(raw_score, 0), 100)
    
    return {"score": round(score, 1)}

@app.post("/predict-text")
async def predict_text(caption: str = Body(..., embed=True)):
    # 1. Tokenize
    # Ensure input is a string
    text = str(caption)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    # 2. Predict
    pred = text_model.predict(padded, verbose=0)[0]
    class_idx = np.argmax(pred)
    confidence = float(pred[class_idx] * 100)
    
    classes = ["Low Engagement", "Average Engagement", "High Engagement"]
    result = classes[class_idx]
    
    return {
        "label": result,
        "confidence": round(confidence, 1),
        "class_id": int(class_idx)
    }

# Mount static files (Frontend)
# This serves index.html at localhost:8000/
app.mount("/", StaticFiles(directory="public", html=True), name="public")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)