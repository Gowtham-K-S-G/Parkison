import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import numpy as np
import cv2

# Load model
print("Loading model...")
model = keras.models.load_model('parkinson_cnn_model.h5')

def preprocess_image(image_path):
    # Same preprocessing as training - just rescale!
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0  # Only rescale - same as training!
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# Test images
test_images = [
    ('static/sample_images/test_parkinson_severe.png', 'Parkinson (Severe)'),
    ('static/sample_images/test_parkinson_mild.png', 'Parkinson (Mild)'),
    ('static/sample_images/test_healthy_clear.png', 'Healthy'),
]

print("\n" + "="*60)
print("TESTING MODEL PREDICTIONS")
print("="*60)

for img_path, expected in test_images:
    if os.path.exists(img_path):
        img = preprocess_image(img_path)
        pred = model.predict(img, verbose=0)[0]
        pred_class = 'Parkinson' if np.argmax(pred) == 1 else 'Healthy'
        print(f"\n{expected}:")
        print(f"   Predicted: {pred_class}")
        print(f"   Healthy: {pred[0]*100:.1f}% | Parkinson: {pred[1]*100:.1f}%")
    else:
        print(f"\nImage not found: {img_path}")

print("\n" + "="*60)
