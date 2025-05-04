import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import imageio
from PIL import Image, UnidentifiedImageError

def _load_image(img_path: str):
    try:
        return Image.open(img_path).convert('RGB')
    except UnidentifiedImageError:
        # Fallback: read via imageio
        arr = imageio.v3.imread(img_path)
        return Image.fromarray(arr).convert('RGB')
# Paths relative to this file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Models', 'Nutrition_Detection', 'server'))

# Load resources once
calorie_df = pd.read_csv(os.path.join(MODEL_DIR, 'calorie_dataset.csv'))
model = load_model(os.path.join(MODEL_DIR, 'nutrition_detection.keras'), compile=False)
with open(os.path.join(MODEL_DIR, 'class_indices.json'), 'r') as f:
    class_map = json.load(f)
index_to_label = {v: k for k, v in class_map.items()}


def predict_and_display_calories(img_path: str) -> dict:
    # Prepare image
    img = _load_image(img_path).resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)

    # Predict
    preds = model.predict(arr)
    idx = int(np.argmax(preds[0]))
    label = index_to_label.get(idx, 'Unknown').strip()

    # Lookup nutrition and return raw strings
    row = calorie_df[calorie_df['Label'].str.strip() == label]
    if not row.empty:
        r = row.iloc[0]
        return {
            'Label': label,
            'Calories': r['Calories'],
            'Carbohydrates': r['Carbohydrates'],
            'Proteins': r['Proteins'],
            'Fats': r['Fats'],
            'Weight': r['Weight']
        }
    else:
        return {'Label': label, 'error': 'No calorie data found'}