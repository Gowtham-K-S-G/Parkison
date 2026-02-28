"""
Generate Sample Spiral Images for Testing
"""

import numpy as np
import cv2
import os
from pathlib import Path

def generate_healthy_spiral(output_path, size=(400, 400)):
    """Generate a healthy (smooth) spiral drawing"""
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    
    # Draw spiral
    center = (size[0] // 2, size[1] // 2)
    
    # Smooth spiral parameters
    num_turns = 3
    points = []
    
    for t in np.linspace(0, num_turns * 2 * np.pi, 500):
        # Archimedean spiral
        r = 5 + t * 15
        x = int(center[0] + r * np.cos(t))
        y = int(center[1] + r * np.sin(t))
        points.append((x, y))
    
    # Draw with varying thickness
    for i in range(len(points) - 1):
        thickness = 2 + int(i / 50)
        cv2.line(img, points[i], points[i + 1], (0, 0, 0), thickness)
    
    cv2.imwrite(output_path, img)
    print(f"✅ Created: {output_path}")


def generate_parkinson_spiral(output_path, size=(400, 400)):
    """Generate a Parkinson's-like spiral with distortions"""
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    
    center = (size[0] // 2, size[1] // 2)
    
    # Spiral parameters with tremor effects
    num_turns = 3
    points = []
    
    np.random.seed(42)
    
    for t in np.linspace(0, num_turns * 2 * np.pi, 500):
        # Archimedean spiral with tremor
        r = 5 + t * 15
        
        # Add tremor (random noise)
        tremor_x = np.random.normal(0, 3)
        tremor_y = np.random.normal(0, 3)
        
        # Add micrographia effect (shrinking at end)
        micrographia = 1 - (t / (num_turns * 2 * np.pi)) * 0.3
        
        x = int(center[0] + (r + tremor_x) * np.cos(t) * micrographia)
        y = int(center[1] + (r + tremor_y) * np.sin(t) * micrographia)
        
        points.append((x, y))
    
    # Draw with irregular thickness
    for i in range(len(points) - 1):
        # Vary thickness randomly (Parkinson's symptom)
        thickness = np.random.randint(1, 4)
        
        # Add some jitter to line path
        if i % 10 == 0:
            jitter = (np.random.randint(-3, 3), np.random.randint(-3, 3))
        else:
            jitter = (0, 0)
        
        pt1 = (points[i][0] + jitter[0], points[i][1] + jitter[1])
        pt2 = points[i + 1]
        
        cv2.line(img, pt1, pt2, (0, 0, 0), thickness)
    
    # Add some extra noise
    noise = np.random.normal(0, 5, (size[0], size[1], 3))
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite(output_path, img)
    print(f"✅ Created: {output_path}")


# Create sample images
static_dir = Path('static/sample_images')
static_dir.mkdir(exist_ok=True)

# Generate healthy spiral
generate_healthy_spiral(str(static_dir / 'healthy_spiral_sample.png'))

# Generate Parkinson's spiral
generate_parkinson_spiral(str(static_dir / 'parkinson_spiral_sample.png'))

print("\n📥 Sample images created!")
print(f"   Location: {static_dir}")
print("\n📤 You can download these images and upload them to test the system at:")
print("   http://127.0.0.1:5000")
