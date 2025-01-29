from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the class labels Excel file
class_labels_df = pd.read_excel('class_labels.xlsx')

# Dictionary to map class indices to person names
class_labels = dict(zip(class_labels_df['Class Index'], class_labels_df['Person Name']))

# Define the allowed file extensions for uploading
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the trained model once when the app starts (to avoid reloading on each request)
model = None
try:
    model = tf.keras.models.load_model('finger_vein_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Function to predict the class of the input image
def predict_finger_vein(image_path):
    if model is None:
        return None, "Model not loaded", 0.0

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224), color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    # Perform prediction
    predictions = model.predict(img_array)
    confidence = float(np.max(predictions)) * 100  # Convert confidence to percentage
    predicted_class = int(np.argmax(predictions))  # Convert to Python int

    # Check confidence threshold
    if confidence < 95.0:
        return None, "Unknown", confidence

    # Fetch person name for the predicted class
    person_name = class_labels.get(predicted_class, "Unknown")

    return predicted_class, person_name, confidence

# Define the main route for the Flask app
@app.route('/')
def index():
    return render_template('index.html')  # Renders the HTML template for uploading the image

# Route for predicting the finger vein class
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)  # Save file to the uploads directory
        file.save(filepath)

        try:
            predicted_class, person_name, confidence = predict_finger_vein(filepath)

            # Optionally delete the file after prediction
            os.remove(filepath)

            return jsonify({
                'predicted_class': predicted_class,
                'person_name': person_name,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error during prediction: {e}")  # Log error in the server console
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file format'}), 400

# Run the app
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')  # Create the uploads directory if it doesn't exist
    app.run(debug=True)