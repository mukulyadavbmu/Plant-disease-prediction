"""
MASTER SCRIPT: Run all training steps sequentially
Run this AFTER step1_preprocess_dataset.py completes.

This will train all 3 models one by one automatically.
Total time: ~2-3 hours
"""

import subprocess
import sys
import time

print("="*80)
print("MASTER TRAINING SCRIPT")
print("="*80)
print("This will train all 3 models sequentially:")
print("  1. MobileNetV2 (~30-40 min)")
print("  2. EfficientNetB0 (~40-50 min)")
print("  3. ResNet50 (~50-60 min)")
print("  4. Evaluate all models (~15 min)")
print("="*80)

scripts = [
    ("Step 2: Training MobileNetV2", "step2_train_mobilenetv2.py"),
    ("Step 3: Training EfficientNetB0", "step3_train_efficientnetb0.py"),
    ("Step 4: Training ResNet50", "step4_train_resnet50.py"),
    ("Step 5: Evaluating All Models", "step5_evaluate_all_models.py")
]

total_start = time.time()

for i, (step_name, script) in enumerate(scripts, 1):
    print(f"\n{'='*80}")
    print(f"{step_name} ({i}/{len(scripts)})")
    print(f"{'='*80}")
    
    step_start = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            check=True,
            capture_output=False
        )
        
        step_time = time.time() - step_start
        minutes = int(step_time // 60)
        seconds = int(step_time % 60)
        
        print(f"\n✓ {step_name} completed in {minutes}m {seconds}s")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR in {step_name}!")
        print(f"Exit code: {e.returncode}")
        print("\nYou can manually run the failed script and then continue with remaining scripts.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)

total_time = time.time() - total_start
total_hours = int(total_time // 3600)
total_minutes = int((total_time % 3600) // 60)

print("\n" + "="*80)
print("ALL TRAINING COMPLETE!")
print("="*80)
print(f"Total time: {total_hours}h {total_minutes}m")
print("\nAll models trained successfully with DIP preprocessing!")
print("Check the models/ folder for saved models and results.")
print("="*80)
