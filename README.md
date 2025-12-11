# InsightLens 👁️

**InsightLens** is a dual-pipeline deep learning tool designed to audit social media content. It analyzes image aesthetics and caption engagement potential to provide creators with a pre-publish quality check.

## 🚀 Key Features
- **Visual Audit:** specific analysis of image technical quality (sharpness, exposure, composition) using **MobileNetV2**.
- **Caption Audit:** specific analysis of text engagement potential using a **Bidirectional LSTM**.
- **Full-Stack Deployment:** Served via **FastAPI** with a vanilla JS/CSS frontend for zero-dependency usage.

---

## 🏗️ System Architecture
InsightLens does **not** use a multimodal approach (like CLIP). Instead, it uses a decoupled architecture to evaluate visual and textual signals independently.

### 1. Vision Module (InsightLens-Vision)
- **Model:** MobileNetV2 (Transfer Learning).
- **Training Data:** KonIQ-10k (10,000 images with human quality ratings).
- **Target:** Regresses a Mean Opinion Score (MOS) normalized to 0-100.
- **Focus:** It judges *technical aesthetics*, not content context. (e.g., A high-quality photo of a wall scores higher than a blurry photo of a celebrity).

### 2. Text Module (InsightLens-Text)
- **Model:** Embedding Layer + Bidirectional LSTM.
- **Training Data:** Twitter Engagement Dataset (~120k tweets).
- **Target:** Classification (Low/Average/High Engagement) based on 'Like' counts.
- **Focus:** It judges *lexical patterns* associated with high engagement, not semantic coherence with the image.

---

## ⚠️ Important Limitations (Transparency)
* **No Contextual Awareness:** The system does not know if the caption matches the image. A picture of a cat with the caption "Nice car" will be scored purely on the photo quality of the cat and the engagement history of the words "Nice car".
* **Twitter Bias:** The text model is trained on Twitter data. It prioritizes "Like-getting" patterns (short, punchy text) over "Comment-getting" patterns (questions).
* **Subjectivity:** "Quality" is subjective. The Image model predicts the average human opinion (MOS), which may not align with specific artistic styles.

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- TensorFlow 2.x

### 1. Setup Environment
```
# Clone the repository
git clone [https://github.com/Mugilan1309/insightlens.git](https://github.com/Mugilan1309/insightlens.git)
cd InsightLens

# Install dependencies
pip install -r requirements.txt
```

2. Run the Application
We use FastAPI to serve the models.


```
python src/api.py
```
Open your browser to http://localhost:8000.


📂 Project Structure
```
InsightLens/
├── models/                  # Pre-trained .h5 artifacts
├── src/
│   ├── api.py               # FastAPI Backend
│   ├── train_image_model.py # Vision Training Pipeline
│   └── train_text_model.py  # NLP Training Pipeline
├── public/                  # Frontend (HTML/JS/CSS)
└── README.md
```
📊 Performance
Vision Model: MAE ~10.6 (on 0-100 scale).

Text Model: Accuracy ~56% (vs 33% random baseline) on 3-class engagement binning.


---