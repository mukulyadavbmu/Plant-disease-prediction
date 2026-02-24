"""
Master script to train all models on RAW images sequentially
Runs: MobileNetV2 → EfficientNetB0 → ResNet50 → Evaluation

Total expected time: ~2-2.5 hours
"""

import subprocess
import time
import sys
import os

print("="*80)
print("RAW IMAGE TRAINING PIPELINE")
print("="*80)
print("\nThis will train 3 models on raw images (224x224, no DIP preprocessing)")
print("Expected total time: 2-2.5 hours")
print("\nModels to train:")
print("  1. MobileNetV2    (~30-40 min)")
print("  2. EfficientNetB0 (~40-50 min)")
print("  3. ResNet50       (~50-60 min)")
print("  4. Evaluation     (~10-15 min)")
print("="*80)
print("\nStarting training automatically in 3 seconds...")
time.sleep(3)

# Scripts to run in order
scripts = [
    ('raw_train_mobilenetv2.py', 'Training MobileNetV2'),
    ('raw_train_efficientnetb0.py', 'Training EfficientNetB0'),
    ('raw_train_resnet50.py', 'Training ResNet50'),
    ('raw_evaluate_all_models.py', 'Evaluating All Models')
]

overall_start = time.time()
completed_steps = []

# Use the correct Python from virtual environment
python_exe = sys.executable
if not os.path.exists(python_exe):
    python_exe = os.path.join('.venv', 'Scripts', 'python.exe')

for i, (script, description) in enumerate(scripts, 1):
    print("\n" + "="*80)
    print(f"Step {i}/4: {description}")
    print("="*80)
    
    step_start = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [python_exe, script],
            check=True,
            capture_output=False,
            text=True
        )
        
        step_time = time.time() - step_start
        completed_steps.append((description, step_time))
        
        print(f"\n✓ Step {i}: {description} completed in {step_time/60:.0f}m {step_time%60:.0f}s")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in step {i}: {description}")
        print(f"Exit code: {e.returncode}")
        print("\nStopping pipeline.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error in step {i}: {description}")
        print(f"Error: {e}")
        print("\nStopping pipeline.")
        sys.exit(1)

# Final summary
total_time = time.time() - overall_start

print("\n" + "="*80)
print("ALL TRAINING COMPLETE!")
print("="*80)
print(f"\nTotal time: {total_time/3600:.1f}h {(total_time%3600)/60:.0f}m")
print("\nCompleted steps:")
for desc, t in completed_steps:
    print(f"  ✓ {desc}: {t/60:.0f}m {t%60:.0f}s")

print("\n" + "="*80)
print("Check the following directories:")
print("  - models_raw/        (trained models)")
print("  - output_plots_raw/  (visualizations)")
print("="*80)
