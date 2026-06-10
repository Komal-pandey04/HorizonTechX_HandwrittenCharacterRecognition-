# =============================================================================
#  main.py  —  FastAPI REST API
#  Handwritten Character Recognition  |  CNN Backend
#
#  Start:   uvicorn main:app --reload --host 0.0.0.0 --port 8000
#  Docs:    http://localhost:8000/docs
# =============================================================================

import os, io, time, logging
import numpy as np, cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Digit Recognition API",
    description="CNN-powered handwritten digit classifier (MNIST, ~99% accuracy).",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "digit_model.h5")
ALLOWED    = {"image/jpeg","image/jpg","image/png","image/bmp","image/webp","image/tiff"}

_model = None
def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model not found at '{MODEL_PATH}'. Run python train.py first.")
        import tensorflow as tf
        t0 = time.perf_counter()
        _model = tf.keras.models.load_model(MODEL_PATH)
        log.info("Model loaded in %.0f ms | params: %s",
                 (time.perf_counter()-t0)*1000, f"{_model.count_params():,}")
    return _model

# ── Schemas ───────────────────────────────────────────────────────────────────
class Top3Item(BaseModel):
    digit: int
    probability: float

class PredictionResponse(BaseModel):
    predicted_digit:   int
    confidence:        float
    confidence_label:  str
    all_probabilities: List[float]
    top3:              List[Top3Item]
    inference_ms:      float
    model_name:        str

class HealthResponse(BaseModel):
    status: str
    model_exists: bool
    message: str

# ── Helpers ───────────────────────────────────────────────────────────────────
def conf_label(p):
    return "High" if p >= 90 else ("Moderate" if p >= 60 else "Low")

def preprocess(raw: bytes) -> np.ndarray:
    """bytes → (1,28,28,1) float32 tensor, MNIST-compatible."""
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(pil)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() > 127:          # white background → invert
        binary = cv2.bitwise_not(binary)
    return (binary.astype("float32") / 255.0).reshape(1, 28, 28, 1)

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Digit Recognition API running.", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
def health():
    exists = os.path.exists(MODEL_PATH)
    return HealthResponse(
        status="ready" if exists else "model_missing",
        model_exists=exists,
        message="Model ready." if exists else "Run python train.py first.",
    )

@app.get("/model/info")
def model_info():
    try:
        m = get_model()
        return {"name": m.name, "total_params": m.count_params(),
                "input_shape": str(m.input_shape), "output_classes": 10}
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"Unsupported type '{file.content_type}'.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file.")
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10 MB).")
    try:
        tensor = preprocess(raw)
    except Exception as e:
        raise HTTPException(422, f"Image processing failed: {e}")
    try:
        m = get_model()
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    t0    = time.perf_counter()
    probs = m.predict(tensor, verbose=0)[0]
    ms    = (time.perf_counter() - t0) * 1000

    digit    = int(np.argmax(probs))
    conf_pct = float(np.max(probs)) * 100
    all_pcts = [round(float(p)*100, 4) for p in probs]
    top3 = [Top3Item(digit=int(i), probability=round(float(probs[i])*100, 4))
            for i in np.argsort(probs)[::-1][:3]]

    log.info("→ digit=%d conf=%.1f%% %s %.1fms", digit, conf_pct,
             conf_label(conf_pct), ms)

    return PredictionResponse(
        predicted_digit=digit, confidence=round(conf_pct, 2),
        confidence_label=conf_label(conf_pct), all_probabilities=all_pcts,
        top3=top3, inference_ms=round(ms, 2), model_name=m.name,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
