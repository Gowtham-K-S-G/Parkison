"""
Training Script for Parkinson's Disease Detection CNN Model
Using Spiral Drawings
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from cnn_model import ParkinsonCNN, preprocess_image
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class TrainParkinsonModel:
    """
    Training pipeline for Parkinson's Disease Detection
    """
    
    def __init__(self, data_dir, img_size=(128, 128), batch_size=32):
        """
        Initialize training parameters
        
        Args:
            data_dir: Path to data directory
            img_size: Target image size
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
    def load_data(self):
        """
        Load and preprocess image data with MORE augmentation
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            # More aggressive augmentation
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Training data
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            color_mode='grayscale',
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation data
        self.validation_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            color_mode='grayscale',
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Classes: {self.train_generator.class_indices}")
        
        return self.train_generator, self.validation_generator
    
    def create_model(self):
        """
        Create and compile the CNN model
        """
        cnn = ParkinsonCNN(input_shape=(self.img_size[0], self.img_size[1], 1))
        self.model = cnn.build_model()
        cnn.compile_model(learning_rate=0.001)
        
        print("Model created successfully!")
        return self.model
    
    def train(self, epochs=30, verbose=1):
        """
        Train the model with fewer epochs to prevent overfitting
        """
        # Callbacks - monitor val_accuracy instead of val_loss
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,  # Reduced patience
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ModelCheckpoint(
                'parkinson_model_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
                mode='max'
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def evaluate(self, test_data_dir):
        """
        Evaluate the model on test data
        """
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=self.img_size,
            color_mode='grayscale',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate
        results = self.model.evaluate(test_generator)
        print("\nTest Results:")
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")
        print(f"AUC: {results[4]:.4f}")
        
        # Predictions
        Y_pred = self.model.predict(test_generator)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = test_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                     target_names=['Healthy', 'Parkinson']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        return results, cm, y_pred, y_true
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Training history plot saved!")
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Healthy', 'Parkinson'],
                    yticklabels=['Healthy', 'Parkinson'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Confusion matrix plot saved!")
    
    def save_model(self, path='parkinson_cnn_model.h5'):
        """
        Save the trained model
        """
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def save_class_names(self):
        """
        Save class names mapping
        """
        class_indices = self.train_generator.class_indices
        class_names = {v: k for k, v in class_indices.items()}
        
        with open('class_names.txt', 'w') as f:
            for idx, name in class_names.items():
                f.write(f"{idx}: {name}\n")
        
        print(f"Class names: {class_names}")


def prepare_sample_data(data_dir):
    """
    Create sample data structure for demonstration
    """
    # Create directories
    for split in ['train', 'test']:
        for label in ['parkinson', 'healthy']:
            os.makedirs(os.path.join(data_dir, split, label), exist_ok=True)
    
    print("Sample data directories created!")
    print(f"Place spiral drawing images in:")
    print(f"  - {data_dir}/train/parkinson/")
    print(f"  - {data_dir}/train/healthy/")
    print(f"  - {data_dir}/test/parkinson/")
    print(f"  - {data_dir}/test/healthy/")


# Main training function
def main():
    """
    Main training pipeline
    """
    print("=" * 60)
    print("Parkinson's Disease Detection - Model Training")
    print("=" * 60)
    
    # Configuration
    DATA_DIR = '../data'
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Create sample data directories
    prepare_sample_data(DATA_DIR)
    
    # Initialize trainer
    trainer = TrainParkinsonModel(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Load data
    print("\n[1/4] Loading data...")
    trainer.load_data()
    
    # Create model
    print("\n[2/4] Creating model...")
    trainer.create_model()
    
    # Train model
    print("\n[3/4] Training model...")
    trainer.train(epochs=EPOCHS)
    
    # Evaluate on test data
    print("\n[4/4] Evaluating model...")
    results, cm, y_pred, y_true = trainer.evaluate(os.path.join(DATA_DIR, 'test'))
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(cm)
    
    # Save model
    trainer.save_model()
    trainer.save_class_names()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
