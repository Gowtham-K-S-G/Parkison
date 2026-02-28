"""
CNN Model Architecture for Parkinson's Disease Detection
Using Spiral Drawings - Simplified for Better Generalization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os


class ParkinsonCNN:
    """
    Convolutional Neural Network for Parkinson's Disease Detection
    from spiral drawings
    """
    
    def __init__(self, input_shape=(128, 128, 1), num_classes=2):
        """
        Initialize the CNN model
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes (2: Parkinson's or Healthy)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['Healthy', 'Parkinson']
        
    def build_model(self):
        """
        Build a SIMPLER CNN architecture for better generalization
        """
        # Use a simpler model to prevent overfitting
        self.model = Sequential([
            # First Convolutional Block - Smaller filters
            Conv2D(16, (3, 3), activation='relu', padding='same', 
                   input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            Dropout(0.2),
            
            # Second Convolutional Block
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.2),
            
            # Third Convolutional Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            # Global Average Pooling instead of Flatten (reduces overfitting)
            GlobalAveragePooling2D(),
            
            # Smaller fully connected layer
            Dense(32, activation='relu'),
            Dropout(0.5),
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function
        """
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(),
                    keras.metrics.Recall(),
                    keras.metrics.AUC()]
        )
        
    def get_model_summary(self):
        """
        Return model summary as string
        """
        return self.model.summary()


def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess a single image for prediction - Same as training
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply adaptive thresholding to enhance features
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Normalize pixel values to [0, 1]
    image = image.astype('float32') / 255.0
    
    # Add channel dimension
    image = np.expand_dims(image, axis=-1)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


def load_trained_model(model_path):
    """
    Load a pre-trained model
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded Keras model
    """
    return keras.models.load_model(model_path)


def predict(model, image_path):
    """
    Make prediction on a single image
    
    Args:
        model: Trained Keras model
        image_path: Path to the image
        
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    image = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(image)
    
    # Get class with highest probability
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    
    # Get all class probabilities
    probabilities = {
        'Healthy': float(predictions[0][0]),
        'Parkinson': float(predictions[0][1])
    }
    
    result = {
        'predicted_class': 'Parkinson' if predicted_class == 1 else 'Healthy',
        'confidence': confidence,
        'probabilities': probabilities
    }
    
    return result


# For testing
if __name__ == "__main__":
    # Create and display model
    cnn = ParkinsonCNN()
    model = cnn.build_model()
    cnn.compile_model()
    print(cnn.get_model_summary())
