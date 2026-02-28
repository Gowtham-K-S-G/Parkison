<<<<<<< HEAD
# Parkinson's Disease Detection System

An AI-powered web application for detecting Parkinson's Disease using spiral drawings and Convolutional Neural Networks (CNNs).

## Overview

This project implements a deep learning-based system to detect Parkinson's Disease from spiral drawings. The system uses a Convolutional Neural Network (CNN) trained on spiral images to classify them as either Parkinson's Disease positive or healthy.

## Features

- **AI-Powered Detection**: Uses advanced CNN architecture for accurate classification
- **Web Interface**: User-friendly Flask-based web application
- **Fast Results**: Instant predictions with confidence scores
- **Non-Invasive**: Simple spiral drawing test - no painful procedures
- **Cost-Effective**: No expensive imaging equipment required

## Project Structure

```
Parkison/
├── app.py                    # Flask web application
├── model/
│   ├── cnn_model.py          # CNN model architecture
│   ├── train.py              # Training script
│   └── requirements.txt      # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css         # Styling
│   ├── js/
│   │   └── main.js           # Frontend JavaScript
│   ├── uploads/              # Uploaded images
│   └── images/               # Static images
├── templates/
│   ├── index.html            # Main page
│   ├── result.html           # Result page
│   ├── about.html            # About page
│   ├── how_it_works.html     # How it works page
│   └── error.html            # Error page
├── data/
│   ├── train/                # Training data
│   │   ├── parkinson/
│   │   └── healthy/
│   └── test/                 # Test data
├── PLAN.md                   # Project plan
└── README.md                 # This file
```

## Installation

1. **Clone the repository**

2. **Create a virtual environment** (optional but recommended)
   
```
bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
```

3. **Install dependencies**
   
```
bash
   cd model
   pip install -r requirements.txt
   
```

## Dataset Preparation

1. Create the following directory structure:
   
```
   data/
   ├── train/
   │   ├── parkinson/     # Put Parkinson's spiral images here
   │   └── healthy/       # Put healthy spiral images here
   └── test/
       ├── parkinson/
       └── healthy/
   
```

2. Place your spiral drawing images in the respective folders

## Training the Model

1. Navigate to the model directory:
   
```
bash
   cd model
   
```

2. Run the training script:
   
```
bash
   python train.py
   
```

3. The trained model will be saved as `parkinson_cnn_model.h5`

## Running the Application

1. Make sure the model file (`parkinson_cnn_model.h5`) is in the root directory

2. Run the Flask application:
   
```
bash
   python app.py
   
```

3. Open your browser and navigate to:
   
```
   http://127.0.0.1:5000
   
```

## Usage

1. **Upload**: Click on the upload area or drag and drop a spiral drawing image
2. **Analyze**: Click the "Analyze Image" button
3. **Results**: View the prediction results with confidence scores

## CNN Model Architecture

The model uses a deep CNN architecture with:
- 4 Convolutional blocks with BatchNormalization
- MaxPooling and Dropout for regularization
- Fully connected layers with softmax output
- Input size: 128x128 grayscale images

## Technical Stack

- **Backend**: Python 3.6+, Flask
- **Deep Learning**: TensorFlow/Keras
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: OpenCV, NumPy

## System Requirements

### Hardware
- Processor: Intel Core i3 or higher
- RAM: 8 GB or higher
- Storage: 10 GB or higher

### Software
- OS: Windows 10 / Linux
- Python 3.6+

## Disclaimer

This system is for educational and research purposes only. The results should not be used as a substitute for professional medical diagnosis. Always consult with a qualified healthcare provider for proper medical advice.

## License

This project is for educational purposes.

## References

- Parkinson's Disease Foundation
- Deep Learning for Medical Diagnosis
- Spiral Drawing Analysis for Parkinson's Detection
=======
# Parkison
An AI-powered web application for detecting Parkinson's Disease using spiral drawings and Convolutional Neural Networks (CNNs).
>>>>>>> 18db36eb97ae3351a9dea54dbfc35ec47f2cabee
