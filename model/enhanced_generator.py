"""
Enhanced Spiral Image Generator for Parkinson's Disease Detection
Generates more realistic spiral drawings with stronger Parkinson symptoms
"""

import numpy as np
import cv2
import os
from pathlib import Path
import random

def generate_enhanced_parkinson_spiral(output_path, seed=None, severity=1.0, size=(400, 400)):
    """
    Generate an enhanced Parkinson's-like spiral with realistic distortions
    
    Args:
        output_path: Path to save the image
        seed: Random seed for reproducibility
        severity: Severity of Parkinson's symptoms (0.0 to 1.0)
        size: Image size (width, height)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    
    center = (size[0] // 2, size[1] // 2)
    
    # Parameters for Parkinson's symptoms
    num_turns = 3.5
    num_points = 600
    
    points = []
    prev_x, prev_y = None, None
    
    for t in np.linspace(0, num_turns * 2 * np.pi, num_points):
        # Archimedean spiral base
        r = 8 + t * 12
        
        # 1. TREMOR EFFECT - Random oscillations
        tremor_intensity = 8 * severity
        tremor_x = np.random.normal(0, tremor_intensity)
        tremor_y = np.random.normal(0, tremor_intensity)
        
        # Low-frequency tremor (more realistic)
        tremor_low_freq = 3 * severity * np.sin(t * 5)
        
        # 2. MICROGRAPHIA - Progressive shrinking
        micrographia = 1 - (t / (num_turns * 2 * np.pi)) * (0.4 * severity)
        
        # 3. BRAKDYKINESIA - Slowness, less defined movements
        bradykinesia = 1 + np.random.uniform(0, 0.3 * severity)
        
        # Calculate position
        x = int(center[0] + (r + tremor_x + tremor_low_freq) * np.cos(t * bradykinesia) * micrographia)
        y = int(center[1] + (r + tremor_y + tremor_low_freq) * np.sin(t * bradykinesia) * micrographia)
        
        # Boundary check
        x = max(10, min(size[0]-10, x))
        y = max(10, min(size[1]-10, y))
        
        points.append((x, y))
        
        # Connect to previous point
        if prev_x is not None:
            # 4. IRREGULAR LINE WIDTH
            thickness = max(1, int(np.random.normal(2, 1.5 * severity)))
            thickness = min(6, thickness)
            
            # 5. JITTER - Unsteady lines
            if random.random() < 0.3 * severity:
                jitter_x = random.randint(-4, 4)
                jitter_y = random.randint(-4, 4)
            else:
                jitter_x, jitter_y = 0, 0
            
            cv2.line(img, 
                    (prev_x + jitter_x, prev_y + jitter_y), 
                    (x, y), 
                    (0, 0, 0), 
                    thickness)
        
        prev_x, prev_y = x, y
    
    # 6. ADD NOISE AND ARTIFACTS
    if severity > 0.5:
        # Add random noise
        noise = np.random.normal(0, 3 * severity, (size[0], size[1], 3))
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        
        # Add some small artifacts (micro-movements)
        for _ in range(int(20 * severity)):
            ax = random.randint(0, size[0]-1)
            ay = random.randint(0, size[1]-1)
            cv2.circle(img, (ax, ay), 1, (200, 200, 200), -1)
    
    # Add some blur for realism
    if severity > 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    cv2.imwrite(output_path, img)
    return output_path


def generate_enhanced_healthy_spiral(output_path, seed=None, size=(400, 400)):
    """
    Generate a healthy (smooth) spiral drawing
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    
    center = (size[0] // 2, size[1] // 2)
    
    num_turns = 3.5
    num_points = 600
    
    points = []
    prev_x, prev_y = None, None
    
    for t in np.linspace(0, num_turns * 2 * np.pi, num_points):
        # Smooth Archimedean spiral
        r = 8 + t * 12
        
        # Slight natural variation (not pathological)
        natural_variation = np.random.normal(0, 0.5)
        
        x = int(center[0] + (r + natural_variation) * np.cos(t))
        y = int(center[1] + (r + natural_variation) * np.sin(t))
        
        points.append((x, y))
        
        if prev_x is not None:
            # Relatively consistent thickness
            thickness = 2 + int(t / 100)
            cv2.line(img, (prev_x, prev_y), (x, y), (0, 0, 0), thickness)
        
        prev_x, prev_y = x, y
    
    cv2.imwrite(output_path, img)
    return output_path


def generate_dataset(base_dir, num_samples=100):
    """
    Generate a dataset of spiral images
    
    Args:
        base_dir: Base directory for saving images
        num_samples: Number of samples per class
    """
    # Create directories
    train_dir = Path(base_dir) / 'train'
    test_dir = Path(base_dir) / 'test'
    
    for split in [train_dir, test_dir]:
        (split / 'parkinson').mkdir(parents=True, exist_ok=True)
        (split / 'healthy').mkdir(parents=True, exist_ok=True)
    
    # Split: 80% train, 20% test
    train_count = int(num_samples * 0.8)
    test_count = num_samples - train_count
    
    print("=" * 60)
    print("GENERATING ENHANCED SPIRAL DATASET")
    print("=" * 60)
    
    # Generate Parkinson's spirals
    print("\n🟢 Generating Parkinson's spirals...")
    for i in range(num_samples):
        # Vary severity
        severity = 0.5 + random.random() * 0.5  # 0.5 to 1.0
        
        if i < train_count:
            path = train_dir / 'parkinson' / f'spiral_pd_{i+1:03d}.png'
        else:
            path = test_dir / 'parkinson' / f'spiral_pd_{i+1:03d}.png'
        
        generate_enhanced_parkinson_spiral(str(path), seed=i*10, severity=severity)
        
        if (i+1) % 20 == 0:
            print(f"   Generated {i+1}/{num_samples}...")
    
    print(f"✅ Generated {num_samples} Parkinson's spirals")
    
    # Generate Healthy spirals
    print("\n🔵 Generating Healthy spirals...")
    for i in range(num_samples):
        if i < train_count:
            path = train_dir / 'healthy' / f'spiral_healthy_{i+1:03d}.png'
        else:
            path = test_dir / 'healthy' / f'spiral_healthy_{i+1:03d}.png'
        
        generate_enhanced_healthy_spiral(str(path), seed=i*10 + 1000)
        
        if (i+1) % 20 == 0:
            print(f"   Generated {i+1}/{num_samples}...")
    
    print(f"✅ Generated {num_samples} Healthy spirals")
    
    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Training set: {train_count * 2} images")
    print(f"  - Parkinson's: {train_count}")
    print(f"  - Healthy: {train_count}")
    print(f"Test set: {test_count * 2} images")
    print(f"  - Parkinson's: {test_count}")
    print(f"  - Healthy: {test_count}")
    print(f"Total: {num_samples * 2} images")
    print(f"\nLocation: {base_dir}")


# Generate dataset
if __name__ == "__main__":
    generate_dataset('../data', num_samples=100)
    print("\n🎉 Enhanced dataset generation complete!")
