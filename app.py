import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

# --- CONFIGURATION ---
MODEL_PATH = "tealeaf.tflite"
# ENSURE THIS ORDER MATCHES YOUR TRAINING DATA EXACTLY
CLASS_NAMES = [
    "Anthracnose",
    "Algal Leaf",
    "Bird Eye Spot",
    "Brown Blight",
    "Gray Blight",
    "Healthy",
    "Red Leaf Spot",
    "White Spot"
]

# --- LOAD MODEL ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/")
def home():
    return {"status": "Online", "model": "Tea Leaf Lens v1"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 2. Resize to 224x224 (Standard MobileNet input)
    image = image.resize((224, 224))

    # 3. Prepare Array
    input_data = np.expand_dims(image, axis=0)

    # IMPORTANT: Check your training!
    # If you used rescaled inputs (1./255), uncomment the next line:
    input_data = np.float32(input_data)

    # 4. Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 5. Format Results
    predicted_index = np.argmax(output_data)
    confidence = float(output_data[0][predicted_index])
    class_name = CLASS_NAMES[predicted_index]

    return {
        "prediction": class_name,
        "confidence": f"{confidence:.2%}",
        "probabilities": {k: f"{v:.4f}" for k, v in zip(CLASS_NAMES, output_data[0])}
    }