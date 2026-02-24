"""
STEP 5: Evaluate and Compare All Models
Evaluates all 4 models (Custom CNN + MobileNetV2 + EfficientNetB0 + ResNet50)
on the test set and generates comprehensive comparison report.

Run after all training steps complete.
Expected time: ~15-20 minutes
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import kagglehub

print("="*80)
print("STEP 5: EVALUATING ALL MODELS")
print("="*80)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IMG_SIZE = 128
BATCH_SIZE = 32

# Get dataset path for test set
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
base_dir = os.path.join(dataset_path, 'New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)')
test_dir = os.path.join(base_dir, 'valid')  # Using valid as test set

# Check preprocessed data
preprocessed_test = 'preprocessed_data/valid'

print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Test directory: {preprocessed_test if os.path.exists(preprocessed_test) else test_dir}")

# ==============================================================================
# LOAD ALL MODELS
# ==============================================================================
print("\n" + "="*80)
print("LOADING ALL TRAINED MODELS")
print("="*80)

models = {}
model_info = {
    'Custom CNN': 'baseline_cnn_model.keras',
    'MobileNetV2': 'models/mobilenetv2.keras',
    'EfficientNetB0': 'models/efficientnetb0.keras',
    'ResNet50': 'models/resnet50.keras'
}

for model_name, model_path in model_info.items():
    if os.path.exists(model_path):
        try:
            models[model_name] = tf.keras.models.load_model(model_path)
            print(f"✓ Loaded: {model_name}")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {str(e)}")
    else:
        print(f"✗ Not found: {model_name} ({model_path})")

if len(models) == 0:
    print("\nERROR: No models found! Please train the models first.")
    exit(1)

print(f"\nTotal models loaded: {len(models)}")

# ==============================================================================
# PREPARE TEST DATA
# ==============================================================================
print("\n" + "="*80)
print("PREPARING TEST DATA")
print("="*80)

# Use preprocessed data if available, otherwise use original
if os.path.exists(preprocessed_test):
    print("Using preprocessed test data...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    use_dir = preprocessed_test
else:
    print("Using original test data (no preprocessing)...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    use_dir = test_dir

test_generator = test_datagen.flow_from_directory(
    use_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = sorted(list(test_generator.class_indices.keys()))
num_classes = len(class_names)

print(f"Test images: {test_generator.samples}")
print(f"Number of classes: {num_classes}")

# ==============================================================================
# EVALUATE EACH MODEL
# ==============================================================================
print("\n" + "="*80)
print("EVALUATING MODELS ON TEST SET")
print("="*80)

results = {}

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    print("-" * 80)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    
    # Get predictions
    print("Generating predictions...")
    y_pred_probs = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes
    
    # Calculate per-class accuracy
    conf_matrix = confusion_matrix(y_true, y_pred)
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    avg_per_class_accuracy = np.mean(per_class_accuracy)
    
    # Get model parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    # Store results
    results[model_name] = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'avg_per_class_accuracy': float(avg_per_class_accuracy),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'predictions': y_pred,
        'true_labels': y_true,
        'confusion_matrix': conf_matrix
    }
    
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Avg Per-Class Accuracy: {avg_per_class_accuracy*100:.2f}%")
    print(f"Total Parameters: {total_params:,}")

# ==============================================================================
# GENERATE COMPARISON REPORT
# ==============================================================================
print("\n" + "="*80)
print("GENERATING COMPARISON REPORT")
print("="*80)

# Create comparison dataframe
comparison_data = []
for model_name, result in results.items():
    comparison_data.append({
        'Model': model_name,
        'Test Accuracy (%)': round(result['test_accuracy'] * 100, 2),
        'Per-Class Accuracy (%)': round(result['avg_per_class_accuracy'] * 100, 2),
        'Test Loss': round(result['test_loss'], 4),
        'Parameters (M)': round(result['total_params'] / 1e6, 2),
        'Trainable Params (M)': round(result['trainable_params'] / 1e6, 2)
    })

df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values('Test Accuracy (%)', ascending=False)

print("\n" + "="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print(df_comparison.to_string(index=False))

# Save comparison table
comparison_csv = 'models/model_comparison.csv'
df_comparison.to_csv(comparison_csv, index=False)
print(f"\nComparison table saved to: {comparison_csv}")

# Save detailed results
results_json = 'models/evaluation_results.json'
results_to_save = {}
for model_name, result in results.items():
    results_to_save[model_name] = {
        'test_accuracy': result['test_accuracy'],
        'test_loss': result['test_loss'],
        'avg_per_class_accuracy': result['avg_per_class_accuracy'],
        'total_params': result['total_params'],
        'trainable_params': result['trainable_params']
    }

with open(results_json, 'w') as f:
    json.dump(results_to_save, f, indent=4)
print(f"Detailed results saved to: {results_json}")

# ==============================================================================
# VISUALIZATION 1: ACCURACY COMPARISON
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Test Accuracy Comparison
ax1 = axes[0, 0]
model_names_list = df_comparison['Model'].tolist()
test_accs = df_comparison['Test Accuracy (%)'].tolist()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = ax1.bar(model_names_list, test_accs, color=colors[:len(model_names_list)])
ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_ylim([85, 100])
ax1.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Model Size Comparison
ax2 = axes[0, 1]
params = df_comparison['Parameters (M)'].tolist()
bars = ax2.bar(model_names_list, params, color=colors[:len(model_names_list)])
ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Parameters (Millions)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 3: Per-Class Accuracy
ax3 = axes[1, 0]
per_class_accs = df_comparison['Per-Class Accuracy (%)'].tolist()
bars = ax3.bar(model_names_list, per_class_accs, color=colors[:len(model_names_list)])
ax3.set_title('Average Per-Class Accuracy', fontsize=14, fontweight='bold')
ax3.set_ylabel('Accuracy (%)', fontsize=12)
ax3.set_ylim([85, 100])
ax3.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 4: Efficiency Score (Accuracy per Million Parameters)
ax4 = axes[1, 1]
efficiency = [acc / params for acc, params in zip(test_accs, params)]
bars = ax4.bar(model_names_list, efficiency, color=colors[:len(model_names_list)])
ax4.set_title('Efficiency Score (Accuracy / Model Size)', fontsize=14, fontweight='bold')
ax4.set_ylabel('Efficiency (Acc% per M params)', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
comparison_plot = 'output_plots/all_models_comparison.png'
plt.savefig(comparison_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nComparison plot saved to: {comparison_plot}")

# ==============================================================================
# VISUALIZATION 2: CONFUSION MATRICES FOR BEST MODEL
# ==============================================================================
best_model_name = df_comparison.iloc[0]['Model']
best_model_cm = results[best_model_name]['confusion_matrix']

plt.figure(figsize=(20, 20))
sns.heatmap(best_model_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name} (Best Model)', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.xticks(rotation=90, fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

cm_plot = f'output_plots/{best_model_name.lower().replace(" ", "_")}_confusion_matrix.png'
plt.savefig(cm_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f"Best model confusion matrix saved to: {cm_plot}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {df_comparison.iloc[0]['Test Accuracy (%)']}%")
print(f"\nAll {len(models)} models evaluated and compared successfully!")
print("\nGenerated files:")
print(f"  - {comparison_csv}")
print(f"  - {results_json}")
print(f"  - {comparison_plot}")
print(f"  - {cm_plot}")
print("\n" + "="*80)
print("PROJECT COMPLETE!")
print("="*80)
print("\nYou now have:")
print(f"  ✓ {len(models)} trained models with DIP preprocessing")
print("  ✓ Comprehensive evaluation results")
print("  ✓ Comparison visualizations")
print("  ✓ Ready for presentation!")
print("="*80)
