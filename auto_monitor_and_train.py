"""
AUTO-MONITOR AND LAUNCH SCRIPT
Monitors preprocessing progress and automatically launches training when done.
"""

import os
import time
import subprocess
import sys

print("="*80)
print("AUTOMATED WORKFLOW MONITOR")
print("="*80)
print("Monitoring preprocessing progress...")
print("Will automatically start training when preprocessing completes.")
print("="*80)

preprocessed_train = 'preprocessed_data/train'
preprocessed_valid = 'preprocessed_data/valid'

# Expected number of classes
expected_classes = 38

def count_completed_classes():
    """Count how many class folders have been preprocessed"""
    train_classes = 0
    valid_classes = 0
    
    if os.path.exists(preprocessed_train):
        train_classes = len([d for d in os.listdir(preprocessed_train) 
                           if os.path.isdir(os.path.join(preprocessed_train, d))])
    
    if os.path.exists(preprocessed_valid):
        valid_classes = len([d for d in os.listdir(preprocessed_valid) 
                           if os.path.isdir(os.path.join(preprocessed_valid, d))])
    
    return train_classes, valid_classes

def is_preprocessing_complete():
    """Check if preprocessing is complete"""
    train_classes, valid_classes = count_completed_classes()
    return train_classes == expected_classes and valid_classes == expected_classes

# Wait for preprocessing to complete
print("\nWaiting for preprocessing to complete...")
last_train = 0
last_valid = 0
check_interval = 30  # Check every 30 seconds

while not is_preprocessing_complete():
    train_classes, valid_classes = count_completed_classes()
    
    # Show progress if changed
    if train_classes != last_train or valid_classes != last_valid:
        print(f"Progress: Train {train_classes}/{expected_classes} | Valid {valid_classes}/{expected_classes} classes")
        last_train = train_classes
        last_valid = valid_classes
    
    time.sleep(check_interval)

# Preprocessing complete!
print("\n" + "="*80)
print("✓ PREPROCESSING COMPLETE!")
print("="*80)

# Wait a bit to ensure all files are written
print("Waiting 10 seconds to ensure all files are saved...")
time.sleep(10)

# Auto-start training
print("\n" + "="*80)
print("AUTO-STARTING MODEL TRAINING")
print("="*80)
print("Training MobileNetV2, EfficientNetB0, and ResNet50...")
print("This will take approximately 2-3 hours.")
print("="*80)

try:
    # Run the training pipeline
    result = subprocess.run([sys.executable, "run_all_training.py"], check=True)
    
    print("\n" + "="*80)
    print("✓ ALL TRAINING COMPLETE!")
    print("="*80)
    print("Check models/ folder for trained models and results.")
    
except subprocess.CalledProcessError as e:
    print(f"\n✗ Training failed with exit code: {e.returncode}")
    print("You can manually run: python run_all_training.py")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\nMonitoring interrupted by user.")
    sys.exit(1)
