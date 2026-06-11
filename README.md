# ✏️ Handwritten Character Recognition — CNN

A complete **full-stack Deep Learning project** — FastAPI backend + professional HTML/CSS/JS frontend — that classifies handwritten digits (0–9) using a CNN trained on MNIST.

---

## 📁 Project Structure

```
Handwritten-Character-Recognition-CNN/
├── backend/
│   ├── train.py           ← Train the CNN (run once)
│   ├── main.py            ← FastAPI REST server
│   └── requirements.txt   ← Python dependencies
├── frontend/
│   └── index.html         ← Complete UI (zero dependencies, single file)
├── model/
│   └── digit_model.h5     ← Saved model (created after training)
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the model (once, ~10–15 min on CPU)
```bash
python train.py
# → saves model/digit_model.h5
# → expected test accuracy: ~99%
```

### 3. Start the API server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# API docs: http://localhost:8000/docs
```

### 4. Open the frontend
```bash
# Just open frontend/index.html in any browser
# Or serve it:
cd ../frontend
python -m http.server 3000
# → http://localhost:3000
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Health check |
| GET  | `/health` | Model status |
| GET  | `/model/info` | Architecture info |
| POST | `/predict` | Classify uploaded image |

### POST /predict — Response
```json
{
  "predicted_digit":   7,
  "confidence":        98.42,
  "confidence_label":  "High",
  "all_probabilities": [0.01, 0.0, 0.12, 0.03, 0.0, 0.0, 0.01, 98.42, 0.0, 0.41],
  "top3": [
    {"digit": 7, "probability": 98.42},
    {"digit": 2, "probability": 0.80},
    {"digit": 9, "probability": 0.41}
  ],
  "inference_ms":  12.4,
  "model_name":    "DigitCNN"
}
```

---

## 🧠 CNN Architecture

```
Input (28×28×1)
 ├─ Conv2D×32 → Conv2D×32 → BatchNorm → MaxPool → Dropout(0.25)
 ├─ Conv2D×64 → Conv2D×64 → BatchNorm → MaxPool → Dropout(0.25)
 ├─ Conv2D×128 → BatchNorm → Dropout(0.25)
 ├─ Flatten
 ├─ Dense×256 → BatchNorm → Dropout(0.4)
 ├─ Dense×128 → Dropout(0.3)
 └─ Dense×10 → Softmax
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Model | TensorFlow 2.x / Keras |
| Backend | FastAPI + Uvicorn |
| Image processing | OpenCV + Pillow |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Dataset | MNIST (auto-downloaded) |

---

## ☁️ Deployment

**Backend** — Railway / Render / Fly.io:
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Commit `model/digit_model.h5` to the repository

**Frontend** — GitHub Pages / Netlify / Vercel:
- Update `const API = "..."` in `index.html` to your backend URL
- Deploy `frontend/` folder (no build step needed)
---
