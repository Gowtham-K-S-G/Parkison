"""
Data Preparation Script for Parkinson's Disease Detection
- Downloads real datasets from public sources
- Generates synthetic spiral drawings for testing
"""

import os
import urllib.request
import zipfile
import numpy as np
import cv2
from pathlib import Path
import shutil

# Configuration
DATA_DIR = Path('../data')
TRAIN_PARKINSON = DATA_DIR / 'train' / 'parkinson'
TRAIN_HEALTHY = DATA_DIR / 'train' / 'healthy'
TEST_PARKINSON = DATA_DIR / 'test' / 'parkinson'
TEST_HEALTHY = DATA_DIR / 'test' / 'healthy'

# Create directories
def create_directories():
    """Create data directories"""
    for dir_path in [TRAIN_PARKINSON, TRAIN_HEALTHY, TEST_PARKINSON, TEST_HEALTHY]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")


def download_real_datasets():
    """
    Download real Parkinson's spiral drawing datasets
    
    Public datasets available:
    1. UCI Parkinson's Dataset - Drawing spirals
    2. Kaggle - Parkinson's Disease Drawings Dataset
    """
    print("\n📥 Downloading Real Datasets...")
    print("=" * 50)
    
    # Note: These are example URLs - you may need to check for current links
    datasets = [
        {
            "name": "UCI Parkinsons Drawing Dataset",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00440/",
            "note": "Check UCI repository for current URL"
        },
        {
            "name": "Kaggle PD Drawing Dataset",
            "url": "https://www.kaggle.com/datasets",
            "note": "Search for Parkinson spiral drawing on Kaggle"
        }
    ]
    
    print("\n📋 Available Public Datasets:")
    print("-" * 50)
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['name']}")
        print(f"   URL: {ds['url']}")
        print(f"   Note: {ds['note']}")
    
    print("\n" + "=" * 50)
    print("💡 To download real data:")
    print("1. Visit the URLs above")
    print("2. Download the datasets")
    print("3. Extract spiral images to:")
    print(f"   - Parkinsons: {TRAIN_PARKINSON}")
    print(f"   - Healthy: {TRAIN_HEALTHY}")
    print("=" * 50)


def generate_synthetic_spirals(output_dir, num_images=20, is_parkinson=False):
    """
    Generate synthetic spiral drawings
    
    Args:
        output_dir: Output directory path
        num_images: Number of images to generate
        is_parkinson: If True, generate Parkinson's-like spirals with distortions
    """
    print(f"\n🎨 Generating {'Parkinson' if is_parkinson else 'Healthy'} spirals...")
    
    for i in range(num_images):
        # Create canvas
        img_size = 400
        img = np.ones((img_size, img_size), dtype=np.uint8) * 255
        
        # Generate spiral parameters
        center_x = img_size // 2 + np.random.randint(-20, 20)
        center_y = img_size // 2 + np.random.randint(-20, 20)
        
        # Base spiral parameters
        num_turns = 3 + np.random.random() * 2
        max_radius = img_size // 2 - 30
        
        # Generate points along spiral
        points = []
        for t in np.linspace(0, num_turns * 2 * np.pi, 500):
            r = (t / (num_turns * 2 * np.pi)) * max_radius
            x = center_x + r * np.cos(t)
            y = center_y + r * np.sin(t)
            points.append((int(x), int(y)))
        
        # Add Parkinson's-like distortions
        if is_parkinson:
            # Add tremor effect (random vibrations)
            tremor_amplitude = np.random.uniform(3, 15)
            points = [
                (x + int(np.random.normal(0, tremor_amplitude)),
                 y + int(np.random.normal(0, tremor_amplitude)))
                for x, y in points
            ]
            
            # Add micrographia (progressive shrinking)
            shrink_factor = np.random.uniform(0.7, 0.95)
            points = [
                (int(center_x + (x - center_x) * shrink_factor * (1 + 0.3 * r / max_radius)),
                 int(center_y + (y - center_y) * shrink_factor * (1 + 0.3 * r / max_radius)))
                for (x, y), r in zip(points, [np.sqrt((x-center_x)**2 + (y-center_y)**2) for x, y in points])
            ]
            
            # Add line width variations
            base_width = np.random.randint(2, 4)
        else:
            base_width = np.random.randint(2, 4)
        
        # Draw the spiral
        for j in range(len(points) - 1):
            x1, y1 = points[j]
            x2, y2 = points[j + 1]
            
            # Vary line width for Parkinson's
            if is_parkinson:
                width = max(1, base_width + int(np.random.normal(0, 2)))
            else:
                width = base_width
            
            cv2.line(img, (x1, y1), (x2, y2), 0, width)
        
        # Add some noise for realism
        if is_parkinson:
            noise_level = np.random.uniform(5, 15)
        else:
            noise_level = np.random.uniform(0, 5)
        
        noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Apply slight blur
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Save image
        filename = f"spiral_{'pd' if is_parkinson else 'healthy'}_{i+1:03d}.png"
        filepath = Path(output_dir) / filename
        cv2.imwrite(str(filepath), img)
    
    print(f"   ✅ Generated {num_images} {'Parkinson' if is_parkinson else 'Healthy'} spiral images")


def generate_all_synthetic_data():
    """Generate all synthetic spiral drawings"""
    print("\n" + "=" * 50)
    print("🎨 GENERATING SYNTHETIC SPIRAL DRAWINGS")
    print("=" * 50)
    
    # Training data
    generate_synthetic_spirals(TRAIN_PARKINSON, num_images=30, is_parkinson=True)
    generate_synthetic_spirals(TRAIN_HEALTHY, num_images=30, is_parkinson=False)
    
    # Test data
    generate_synthetic_spirals(TEST_PARKINSON, num_images=10, is_parkinson=True)
    generate_synthetic_spirals(TEST_HEALTHY, num_images=10, is_parkinson=False)
    
    print("\n" + "=" * 50)
    print("✅ Synthetic data generation complete!")
    print("=" * 50)
    print(f"\n📁 Data saved to:")
    print(f"   Training: {TRAIN_PARKINSON.parent}")
    print(f"   Testing:  {TEST_PARKINSON.parent}")


def list_data_summary():
    """List summary of available data"""
    print("\n📊 DATA SUMMARY")
    print("=" * 50)
    
    def count_images(dir_path):
        if dir_path.exists():
            return len(list(dir_path.glob('*.png'))) + len(list(dir_path.glob('*.jpg')))
        return 0
    
    train_park = count_images(TRAIN_PARKINSON)
    train_healthy = count_images(TRAIN_HEALTHY)
    test_park = count_images(TEST_PARKINSON)
    test_healthy = count_images(TEST_HEALTHY)
    
    print(f"\nTraining Data:")
    print(f"   Parkinson's: {train_park} images")
    print(f"   Healthy:      {train_healthy} images")
    print(f"   Total:        {train_park + train_healthy} images")
    
    print(f"\nTest Data:")
    print(f"   Parkinson's: {test_park} images")
    print(f"   Healthy:      {test_healthy} images")
    print(f"   Total:        {test_park + test_healthy} images")
    
    total = train_park + train_healthy + test_park + test_healthy
    if total == 0:
        print("\n⚠️  No data available. Run this script to generate synthetic data!")
    else:
        print(f"\n✅ Total: {total} images available for training")


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🧠 PARKINSON'S DISEASE DETECTION - DATA PREPARATION")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Show download options for real datasets
    download_real_datasets()
    
    # Ask user what to do next
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("Option 1: Generate synthetic spiral drawings for testing")
    print("   Run: python generate_synthetic_data()")
    print("\nOption 2: Download real datasets")
    print("   Visit the URLs provided above")
    print("   Place downloaded images in data/train/ folders")
    print("=" * 60)
    
    # Generate synthetic data
    generate_all_synthetic_data()
    
    # Show summary
    list_data_summary()
    
    print("\n" + "=" * 60)
    print("🎉 DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train the model: python train.py")
    print("2. Run the app: python ../app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
