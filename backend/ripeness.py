import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Paths
MODEL_PATH = r'D:\Mini_Project\Models\Ripeness_Detection\server\ripeness_detection.keras'
CLASS_MAP_PATH = r'D:\Mini_Project\Models\Ripeness_Detection\server\classes.json'
IMG_WIDTH, IMG_HEIGHT = 128, 128  

# Load model and class indices
model = load_model(MODEL_PATH)
with open(CLASS_MAP_PATH, 'r') as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

def predict_ripeness(img_path):
    if not os.path.exists(img_path):
        return "Invalid Image Path", 0.0

    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = index_to_class[predicted_index]
    confidence = float(np.max(predictions[0]))

    return predicted_class, confidence
