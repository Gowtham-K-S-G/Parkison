"""
Flask Web Application for Parkinson's Disease Detection
Using Spiral Drawings and CNN
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow import keras
import tensorflow as tf

# Import model utilities
from model.cnn_model import preprocess_image, predict as model_predict

# Configuration
app = Flask(__name__)
app.secret_key = 'parkinson_detection_secret_key'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
MODEL_PATH = 'model/parkinson_cnn_model.h5'
IMG_SIZE = (128, 128)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable
model = None


def allowed_file(filename):
    """
    Check if file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """
    Load the trained CNN model
    """
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
            print("Please train the model first using model/train.py")
            model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


def preprocess_uploaded_image(image_path):
    """
    Preprocess uploaded image for prediction - Same as training!
    Training uses ImageDataGenerator with rescale=1./255 only
    """
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Resize image
    image = cv2.resize(image, IMG_SIZE)
    
    # Normalize pixel values to [0, 1] - Same as training (rescale=1./255)!
    image = image.astype('float32') / 255.0
    
    # Add channel dimension
    image = np.expand_dims(image, axis=-1)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


def make_prediction(image_path):
    """
    Make prediction on the uploaded image
    """
    # Preprocess image
    image = preprocess_uploaded_image(image_path)
    
    # Make prediction
    predictions = model.predict(image, verbose=0)
    
    # Get class with highest probability
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    
    # Determine result
    if predicted_class == 1:
        result = "Parkinson's Disease Detected"
        result_class = "parkinson"
    else:
        result = "No Parkinson's Disease Detected"
        result_class = "healthy"
    
    # Get all probabilities
    probabilities = {
        'healthy': {
            'probability': float(predictions[0][0]),
            'percentage': f"{predictions[0][0] * 100:.2f}%"
        },
        'parkinson': {
            'probability': float(predictions[0][1]),
            'percentage': f"{predictions[0][1] * 100:.2f}%"
        }
    }
    
    return {
        'result': result,
        'result_class': result_class,
        'confidence': f"{confidence * 100:.2f}%",
        'probabilities': probabilities
    }


# Routes
@app.route('/')
def index():
    """
    Home page - upload form
    """
    return render_template('index.html')


@app.route('/about')
def about():
    """
    About page
    """
    return render_template('about.html')


@app.route('/how-it-works')
def how_it_works():
    """
    How it works page
    """
    return render_template('how_it_works.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and prediction
    """
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure the model is trained and saved.'
        }), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'error': 'No file selected'
        }), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            # Add timestamp to filename to avoid duplicates
            import time
            timestamp = int(time.time())
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = make_prediction(filepath)
            
            # Add image path to result
            result['image_path'] = f"static/uploads/{filename}"
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'error': f'Error processing image: {str(e)}'
            }), 500
    
    return jsonify({
        'error': 'Invalid file type. Please upload an image file.'
    }), 400


@app.route('/result')
def result():
    """
    Result display page
    """
    return render_template('result.html')


@app.route('/sample-images')
def sample_images():
    """
    Sample images for testing
    """
    sample_dir = 'static/sample_images'
    samples = []
    
    if os.path.exists(sample_dir):
        for filename in os.listdir(sample_dir):
            if allowed_file(filename):
                samples.append(f"static/sample_images/{filename}")
    
    return jsonify(samples)


@app.route('/model-info')
def model_info():
    """
    Get model information
    """
    if model is None:
        return jsonify({
            'status': 'Model not loaded',
            'message': 'Please train the model first'
        })
    
    return jsonify({
        'status': 'Model loaded',
        'model_type': 'CNN (Convolutional Neural Network)',
        'input_shape': '128x128 grayscale',
        'classes': ['Healthy', 'Parkinson']
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """404 Error handler"""
    return render_template('error.html', error_code=404, 
                           error_message="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    """500 Error handler"""
    return render_template('error.html', error_code=500, 
                           error_message="Internal server error"), 500


# Main entry point
if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    load_model()
    
    # Run the app
    print("\n" + "="*60)
    print("Parkinson's Disease Detection System")
    print("="*60)
    print("Starting Flask server...")
    print("Access the application at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
