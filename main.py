from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your trained model
MODEL_PATH = "api/model.h5"
model = load_model(MODEL_PATH)


# Preprocess the image
def preprocess_image(image):
    image = image.resize((256, 256))  # Adjust size according to model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    # Map the prediction to the corresponding class name
    class_names = ["Class1", "Class2", "Class3"]  # Update with your actual classes
    result = {
        'predicted_class': class_names[predicted_class],
        'confidence': f"{confidence:.2f}%"
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
