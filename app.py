from flask import Flask, request, render_template, jsonify
import joblib
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'poultry_disease_image_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Class indices (should match the training order)
class_indices = {
    0: 'Coccidiosis',
    1: 'Healthy',
    2: 'New Castle Disease',
    3: 'Salmonella'
}

IMG_SIZE = 224  # Change to your model's expected input size

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files.get('imagefile')
        if file and file.filename:
            image = Image.open(file.stream)
            img_arr = preprocess_image(image)
            pred = model.predict(img_arr)
            pred_class = np.argmax(pred, axis=1)[0]
            prediction = class_indices.get(pred_class, 'Unknown')
    return render_template('index.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)
        img = preprocess_image(Image.open(file_path))
        preds = model.predict(img)
        pred_class = np.argmax(preds, axis=1)[0]
        result = class_indices.get(pred_class, 'Unknown')
        os.remove(file_path)
        return jsonify({'prediction': result})
    return jsonify({'error': 'File processing error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
