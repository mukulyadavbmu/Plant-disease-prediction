"""
STEP 1: Dataset Preprocessing Pipeline
Applies advanced DIP techniques to all images and saves them to disk.
This needs to be run ONCE before training any models.

DIP Techniques Applied:
1. Bilateral Filter - Edge-preserving noise reduction
2. CLAHE (LAB color space) - Contrast enhancement
3. Unsharp Masking - Edge sharpening

Run time: ~1-2 hours (one-time only)
"""

import os
import cv2
import numpy as np
from PIL import Image
import kagglehub
from tqdm import tqdm
import shutil

print("="*80)
print("STEP 1: PREPROCESSING DATASET WITH ADVANCED DIP TECHNIQUES")
print("="*80)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IMG_SIZE = 128

# Get dataset path
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
base_dir = os.path.join(dataset_path, 'New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)')
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

# Output directories for preprocessed images
preprocessed_base = 'preprocessed_data'
preprocessed_train = os.path.join(preprocessed_base, 'train')
preprocessed_valid = os.path.join(preprocessed_base, 'valid')

print(f"\nSource dataset: {base_dir}")
print(f"Output directory: {preprocessed_base}")

# ==============================================================================
# ADVANCED DIP PREPROCESSING FUNCTION
# ==============================================================================
def advanced_preprocessing(image_path, output_path, target_size=128):
    """
    Load image, apply DIP techniques, and save to output path.
    
    Args:
        image_path: Path to original image
        output_path: Path to save preprocessed image
        target_size: Resize dimension (default 128x128)
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load {image_path}")
            return False
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, (target_size, target_size))
        
        # Step 1: Bilateral Filter (denoise while preserving edges)
        bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Step 2: CLAHE in LAB color space (enhance contrast)
        lab = cv2.cvtColor(bilateral, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        clahe_result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        # Step 3: Unsharp Masking (sharpen edges)
        gaussian = cv2.GaussianBlur(clahe_result, (0, 0), 2.0)
        unsharp = cv2.addWeighted(clahe_result, 1.5, gaussian, -0.5, 0)
        
        # Ensure values are in valid range
        unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
        
        # Convert RGB back to BGR for saving with OpenCV
        unsharp_bgr = cv2.cvtColor(unsharp, cv2.COLOR_RGB2BGR)
        
        # Save preprocessed image
        cv2.imwrite(output_path, unsharp_bgr)
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

# ==============================================================================
# PREPROCESSING PIPELINE
# ==============================================================================
def preprocess_directory(source_dir, target_dir, split_name):
    """
    Process all images in a directory and save to target directory.
    Maintains the same folder structure (class folders).
    """
    print(f"\n{'='*80}")
    print(f"Processing {split_name} set...")
    print(f"{'='*80}")
    
    # Get all class folders
    class_folders = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])
    print(f"Found {len(class_folders)} classes")
    
    total_processed = 0
    total_failed = 0
    
    # Process each class folder
    for class_name in tqdm(class_folders, desc="Classes"):
        source_class_dir = os.path.join(source_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)
        
        # Create target class directory
        os.makedirs(target_class_dir, exist_ok=True)
        
        # Get all images in this class
        image_files = [f for f in os.listdir(source_class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Process each image
        for img_file in image_files:
            source_path = os.path.join(source_class_dir, img_file)
            target_path = os.path.join(target_class_dir, img_file)
            
            # Skip if already processed
            if os.path.exists(target_path):
                total_processed += 1
                continue
            
            # Process and save
            if advanced_preprocessing(source_path, target_path, IMG_SIZE):
                total_processed += 1
            else:
                total_failed += 1
    
    print(f"\n{split_name} set complete:")
    print(f"  ✓ Processed: {total_processed} images")
    if total_failed > 0:
        print(f"  ✗ Failed: {total_failed} images")
    
    return total_processed, total_failed

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    print("\n" + "="*80)
    print("STARTING PREPROCESSING PIPELINE")
    print("="*80)
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"DIP Techniques: Bilateral Filter → CLAHE (LAB) → Unsharp Masking")
    
    # Create base directory
    os.makedirs(preprocessed_base, exist_ok=True)
    
    # Check if preprocessing already done
    if os.path.exists(preprocessed_train) and os.path.exists(preprocessed_valid):
        train_classes = len(os.listdir(preprocessed_train))
        valid_classes = len(os.listdir(preprocessed_valid))
        
        if train_classes > 0 and valid_classes > 0:
            print("\n" + "="*80)
            print("PREPROCESSED DATA ALREADY EXISTS!")
            print("="*80)
            print(f"Found preprocessed data in: {preprocessed_base}")
            print(f"  - Training classes: {train_classes}")
            print(f"  - Validation classes: {valid_classes}")
            
            response = input("\nDo you want to re-preprocess? (yes/no): ").strip().lower()
            if response != 'yes':
                print("\nSkipping preprocessing. Using existing preprocessed data.")
                print("You can now proceed to train the models using:")
                print("  - step2_train_mobilenetv2.py")
                print("  - step3_train_efficientnetb0.py")
                print("  - step4_train_resnet50.py")
                exit(0)
            else:
                print("\nRemoving existing preprocessed data...")
                shutil.rmtree(preprocessed_base)
                os.makedirs(preprocessed_base, exist_ok=True)
    
    # Process training set
    train_processed, train_failed = preprocess_directory(train_dir, preprocessed_train, "Training")
    
    # Process validation set
    valid_processed, valid_failed = preprocess_directory(valid_dir, preprocessed_valid, "Validation")
    
    # Calculate total time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    # Final summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"\nTotal images processed: {train_processed + valid_processed}")
    print(f"  - Training: {train_processed} images")
    print(f"  - Validation: {valid_processed} images")
    
    if train_failed + valid_failed > 0:
        print(f"\nWarning: {train_failed + valid_failed} images failed to process")
    
    print(f"\nPreprocessed data saved to: {preprocessed_base}/")
    
    # Auto-start training pipeline
    print("\n" + "="*80)
    print("AUTO-STARTING MODEL TRAINING PIPELINE")
    print("="*80)
    print("Training all 3 models automatically...")
    print("This will take approximately 2-3 hours.")
    print("="*80)
    
    import subprocess
    import sys
    
    try:
        print("\nLaunching run_all_training.py...")
        subprocess.run([sys.executable, "run_all_training.py"], check=True)
    except Exception as e:
        print(f"\nError launching training: {e}")
        print("\nYou can manually run: python run_all_training.py")
    
    print("\n" + "="*80)
