"""
STEP 2 - an API that runs our trained model.

You send it a picture  ->  it answers "cat, 93 %".

Run it:   python step2_predict_api.py
Test it:  curl.exe -F "file=@../app/ui/samples/cat_1.png" http://127.0.0.1:5001/predict
          (or use step3_simple_ui.py)
"""
import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, request
from PIL import Image

# ---------------------------------------------------------------- 1. load the model ONCE, at start-up
# The model was trained on Day 2 and saved in the <repo>/models folder.
MODEL_FILE = Path(__file__).resolve().parents[2] / "models" / "model_int8.onnx"
model = ort.InferenceSession(str(MODEL_FILE))

LABELS = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]


# ---------------------------------------------------------------- 2. picture -> numbers
# This MUST be exactly what we did during training, otherwise the model gets confused.
def picture_to_numbers(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")    # open the picture, make sure it's colour
    img = img.resize((96, 96), Image.BILINEAR)                  # same size as in training
    x = np.asarray(img, dtype=np.float32) / 255.0               # pixels 0..255  ->  0..1
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)    # same normalisation as in training
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)                                    # (height, width, colour) -> (colour, height, width)
    return x[np.newaxis, ...].astype(np.float32)                # add a "batch" dimension: 1 picture


# ---------------------------------------------------------------- 3. the API
app = Flask(__name__)


@app.get("/")
def home():
    return {"message": "Send a picture with POST /predict"}


@app.post("/predict")
def predict():
    if "file" not in request.files:                             # friendly error if no picture was sent
        return {"error": "Please send a picture in the field 'file'"}, 400

    x = picture_to_numbers(request.files["file"].read())

    scores = model.run(None, {"input": x})[0][0]                # 10 raw scores, one per class
    probs = np.exp(scores - scores.max())                       # softmax: scores -> probabilities
    probs = probs / probs.sum()                                 # (they now add up to 1)

    best = int(np.argmax(probs))                                # the class with the highest probability
    return {
        "label": LABELS[best],
        "confidence": round(float(probs[best]), 3),
        "all_probabilities": {LABELS[i]: round(float(p), 3) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    app.run(port=5001)
