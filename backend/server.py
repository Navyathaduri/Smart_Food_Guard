import os
# Suppress TF info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify
from flask_cors import CORS
import nutrition
import shelflife
import ripeness
import spoilage 
# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

# Create app: serve everything in frontend/ as static
app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path='',
    template_folder=FRONTEND_DIR
)
CORS(app)

# Routes for HTML pages
@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/nutrition.html')
def serve_nutrition():
    return app.send_static_file('nutrition.html')

@app.route('/ripeness.html')
def serve_ripeness():
    return app.send_static_file('ripeness.html')

@app.route('/shelflife.html')
def serve_shelflife():
    return app.send_static_file('shelflife.html')

@app.route('/spoilage.html')
def serve_spoilage():
    return app.send_static_file('spoilage.html')

# Nutrition API endpoint
@app.route('/predict_and_display_calories', methods=['POST'])
def predict_and_display_calories_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save upload
    temp_dir = os.path.join(BASE_DIR, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    # Run prediction
    try:
        result = nutrition.predict_and_display_calories(temp_path)
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 400
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return jsonify(result)


@app.route('/predict_shelflife', methods=['POST'])
def predict_shelf_life_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_dir = os.path.join(BASE_DIR, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        result = shelflife.predict_shelflife(temp_path)
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 400
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return jsonify(result)

@app.route('/predict_ripeness', methods=['POST'])
def predict_ripeness_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file
    temp_dir = os.path.join(BASE_DIR, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        predicted_class, confidence = ripeness.predict_ripeness(temp_path)
        result = {
            'ripenessStatus': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return jsonify(result)

@app.route('/predict_spoilage', methods=['POST'])
def predict_spoilage_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file
    temp_dir = os.path.join(BASE_DIR, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        result, confidence = spoilage.predict_binary_spoilage(temp_path)
        result_data = {
            'spoilageStatus': result,
            'confidence': confidence
        }
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return jsonify(result_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)