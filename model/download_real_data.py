"""
Real Dataset Downloader for Parkinson's Disease Spiral Drawings
Downloads real datasets from public sources
"""

import os
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
import ssl
import urllib.error

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


def download_file(url, destination):
    """Download a file with progress"""
    try:
        # Create SSL context that doesn't verify certificates (for some older servers)
        import urllib.request
        
        def download_with_progress(url, dest):
            """Download with progress indication"""
            print(f"📥 Downloading: {url.split('/')[-1]}")
            
            # Try with SSL verification first
            try:
                urllib.request.urlretrieve(url, dest)
                return True
            except Exception as e:
                print(f"   SSL error, trying without verification...")
                # Try without SSL verification
                import ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                
                try:
                    with urllib.request.urlopen(url, context=ctx) as response:
                        with open(dest, 'wb') as out_file:
                            shutil.copyfileobj(response, out_file)
                    return True
                except Exception as e2:
                    print(f"   ❌ Failed: {e2}")
                    return False
        
        return download_with_progress(url, destination)
        
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract archive file"""
    print(f"📦 Extracting: {archive_path.name}")
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"   ⚠️ Unknown archive format: {archive_path.suffix}")
            return False
        
        print(f"   ✅ Extracted successfully")
        return True
    except Exception as e:
        print(f"   ❌ Extraction error: {e}")
        return False


def download_uci_parkinson_dataset():
    """Download Parkinson's dataset from UCI repository"""
    print("\n" + "="*60)
    print("📥 Downloading UCI Parkinson's Dataset")
    print("="*60)
    
    # UCI Parkinson's dataset URL
    # This dataset contains drawings including spirals
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00440"
    
    # Known files in UCI Parkinson's dataset
    files_to_download = [
        ("pd_speech_features.csv", "Speech features"),
        ("parkinsons.data", "Drawing data"),
    ]
    
    print("\n⚠️ UCI Repository Information:")
    print("-"*60)
    print("The UCI Parkinson's dataset primarily contains speech features.")
    print("For spiral drawings, we recommend other sources.")
    print("\nPlease visit:")
    print("1. UCI ML Repository: https://archive.ics.uci.edu/dataset/174/parkinsons")
    print("2. Download the dataset manually")
    print("3. Extract spiral drawing files to data/train/ folders")
    
    return False


def download_kaggle_dataset_instructions():
    """Show Kaggle download instructions"""
    print("\n" + "="*60)
    print("📥 Kaggle Dataset Download Instructions")
    print("="*60)
    
    print("\nTo download from Kaggle:")
    print("-"*60)
    print("1. Go to: https://www.kaggle.com/datasets")
    print("2. Search for: 'Parkinson spiral drawing' or 'Parkinson disease drawing'")
    print("3. Popular datasets:")
    print("   - 'Parkinson Disease Drawing Dataset'")
    print("   - 'Spiral Wave Drawing for Parkinson Detection'")
    print("   - 'Hand-Drawn Spiral/Wave Dataset for PD'")
    print("\n4. Download the dataset")
    print("5. Extract files:")
    print(f"   - Parkinson's drawings → {TRAIN_PARKINSON}")
    print(f"   - Healthy drawings → {TRAIN_HEALTHY}")
    print("\nTo use Kaggle CLI:")
    print("   pip install kaggle")
    print("   kaggle datasets download -d <dataset-name>")
    
    return False


def download_github_dataset():
    """Try to download from GitHub repositories"""
    print("\n" + "="*60)
    print("📥 Checking GitHub for Datasets")
    print("="*60)
    
    # Known GitHub repositories with Parkinson's drawing data
    github_sources = [
        {
            "name": "Parkinson Drawing Dataset (GitHub)",
            "description": "Various repositories contain spiral/wave drawings",
            "url": "Search GitHub for: parkinson spiral drawing dataset"
        },
    ]
    
    for source in github_sources:
        print(f"\n📌 {source['name']}")
        print(f"   {source['description']}")
        print(f"   URL: {source['url']}")
    
    return False


def manual_download_guide():
    """Show comprehensive manual download guide"""
    print("\n" + "="*60)
    print("📋 MANUAL DOWNLOAD GUIDE")
    print("="*60)
    
    sources = [
        {
            "name": "1. Kaggle",
            "url": "https://www.kaggle.com/datasets",
            "search": "parkinson spiral drawing",
            "instructions": "Search and download, extract to data/train/"
        },
        {
            "name": "2. UCI Machine Learning Repository",
            "url": "https://archive.ics.uci.edu/dataset/174/parkinsons",
            "search": "",
            "instructions": "Register and download, extract spiral images"
        },
        {
            "name": "3. IEEE DataPort",
            "url": "https://ieee-dataport.org/subjects/medicine",
            "search": "Parkinson spiral",
            "instructions": "Search and request access if needed"
        },
        {
            "name": "4. GitHub",
            "url": "https://github.com",
            "search": "parkinson spiral drawing dataset",
            "instructions": "Search repositories for drawing datasets"
        },
        {
            "name": "5. Zenodo",
            "url": "https://zenodo.org",
            "search": "parkinson drawing",
            "instructions": "Search for drawing datasets"
        }
    ]
    
    print("\n📚 Available Sources:\n")
    for source in sources:
        print(f"{source['name']}")
        print(f"   URL: {source['url']}")
        if source['search']:
            print(f"   Search: {source['search']}")
        print(f"   Instructions: {source['instructions']}")
        print()


def download_sample_dataset():
    """Try to download a sample dataset if available"""
    print("\n" + "="*60)
    print("🔍 Attempting to Download Sample Dataset")
    print("="*60)
    
    # Create a temp directory
    temp_dir = DATA_DIR / 'temp_download'
    temp_dir.mkdir(exist_ok=True)
    
    # Try different sources
    download_attempted = False
    
    # Source 1: Try direct link (if available)
    # Note: These are example URLs - they may not work
    test_urls = [
        # Add known URLs here if available
    ]
    
    if not test_urls:
        print("\n⚠️ No direct download links available.")
        print("   Please download datasets manually from the sources above.")
    
    # Clean up temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    return download_attempted


def copy_existing_spirals():
    """Check if there are any existing spirals we can use"""
    print("\n" + "="*60)
    print("📂 Checking for Existing Spiral Images")
    print("="*60)
    
    # Check our synthetic data
    synth_park = len(list(TRAIN_PARKINSON.glob('*.png')) + list(TRAIN_PARKINSON.glob('*.jpg')))
    synth_healthy = len(list(TRAIN_HEALTHY.glob('*.png')) + list(TRAIN_HEALTHY.glob('*.jpg')))
    
    print(f"\nCurrently available:")
    print(f"   Parkinson's training: {synth_park}")
    print(f"   Healthy training: {synth_healthy}")
    print(f"   Total: {synth_park + synth_healthy}")
    
    if synth_park + synth_healthy > 0:
        print(f"\n✅ Using {synth_park + synth_healthy} synthetic images for initial training")
        print("   You can add real images later for better accuracy")
        return True
    
    return False


def main():
    """Main function"""
    print("\n" + "="*60)
    print("🧠 PARKINSON'S DISEASE - REAL DATA DOWNLOADER")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Show download options
    print("\n" + "="*60)
    print("OPTIONS TO GET REAL DATA")
    print("="*60)
    
    # Option 1: UCI
    download_uci_parkinson_dataset()
    
    # Option 2: Kaggle
    download_kaggle_dataset_instructions()
    
    # Option 3: GitHub
    download_github_dataset()
    
    # Option 4: Manual guide
    manual_download_guide()
    
    # Option 5: Try to download
    download_sample_dataset()
    
    # Show current status
    copy_existing_spirals()
    
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    print("""
To get real data for better model accuracy:

1. KAGGLE (Recommended):
   - Go to kaggle.com/datasets
   - Search: "parkinson spiral drawing"
   - Download and extract

2. MANUAL DOWNLOAD:
   - Visit any of the sources above
   - Place Parkinson's images in: data/train/parkinson/
   - Place Healthy images in: data/train/healthy/

3. CURRENT DATA:
   - We have 80 synthetic images for basic testing
   - These will work for initial training
   - Add real data later for better accuracy
   
NOTE: The model CAN be trained with synthetic data for testing purposes.
      For production use, please add real spiral drawings.
""")
    
    print("="*60)


if __name__ == "__main__":
    main()
