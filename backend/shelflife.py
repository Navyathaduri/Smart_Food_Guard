import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model(r'D:\Mini_Project\Models\Shelflife_Detection\server\Shelf_life_prediction.keras')  

# Load class name mapping from JSON file
with open(r'D:\Mini_Project\Models\Shelflife_Detection\server\classes.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the mapping: index to class name
index_to_class = {v: k for k, v in class_indices.items()}

# Shelf life mapping
shelf_life_mapping = {
    'unripe apple': '7',
    'freshapples': '4',
    'rottenapples': '0',
    'unripe banana': '5',
    'freshbanana': '3',
    'rottenbanana': '0',
    'unripe orange': '10',
    'freshoranges': '6',
    'rottenoranges': '0'
}

# Image size used during training
IMG_SIZE = 224

def predict_shelflife(image_path):
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = index_to_class[predicted_index]
    confidence = float(np.max(predictions[0]))
    shelf_life = shelf_life_mapping.get(predicted_class, "Unknown")

    return {
        'filename': os.path.basename(image_path),
        'predicted_class': predicted_class,
        'confidence': round(confidence, 2),
        'shelf_life_days': shelf_life
    }

# Test
test_image_path = r'D:\Mini_Project\Models\Shelflife_Detection\server\test_images\img6.png'  # Update with your actual test image
predict_shelflife(test_image_path) 