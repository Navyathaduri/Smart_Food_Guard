import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_binary_spoilage(img_path, model_path=r'D:\Mini_Project\Models\Spoilage_Detection\server\Mini_Spoilage_Detection_Model.keras'):
    # Load model
    model = load_model(model_path)

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(150, 150))  # Match model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    
    # Return class based on threshold
    if prediction < 0.5:
        return "Fresh", float(prediction)
    else:
        return "Spoiled", float(prediction)


