import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = 96  # your model input size
MODEL_PATH = os.path.join("models", "best_emotion_model.h5")
CLASSES_PATH = "classes.json"

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Load classes
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Classes file not found: {CLASSES_PATH}")

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

inv_classes = {int(v): k for k, v in classes.items()}


def _preprocess(pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(pil_img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_emotion(image_path):
    """Predict from path"""
    pil_img = Image.open(image_path)
    arr = _preprocess(pil_img)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    return inv_classes[idx]


def predict_emotion_from_pil(pil_img):
    """Predict from PIL image"""
    arr = _preprocess(pil_img)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    return inv_classes[idx]
