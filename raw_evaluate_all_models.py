"""
Evaluate all models trained on RAW images and compare with Custom CNN
Generates comprehensive comparison report

Compares:
- Custom CNN (95.69% on raw data) 
- MobileNetV2 (trained on raw 224x224)
- EfficientNetB0 (trained on raw 224x224)
- ResNet50 (trained on raw 224x224)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import kagglehub

print("="*80)
print("EVALUATING ALL MODELS - RAW IMAGE COMPARISON")
print("="*80)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BATCH_SIZE = 32
NUM_CLASSES = 38

# Get dataset path
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
base_dir = os.path.join(dataset_path, 'New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)')
valid_dir = os.path.join(base_dir, 'valid')

models_dir = 'models_raw'
plots_dir = 'output_plots_raw'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Test directory: {valid_dir}")

# ==============================================================================
# LOAD MODELS
# ==============================================================================
print("\n" + "="*80)
print("LOADING ALL TRAINED MODELS")
print("="*80)

models_to_evaluate = {}

# Load Custom CNN (original baseline)
try:
    models_to_evaluate['Custom CNN'] = {
        'model': load_model('baseline_cnn_model.keras'),
        'img_size': 128,
        'preprocess': lambda x: x / 255.0
    }
    print("✓ Loaded: Custom CNN (baseline, 128x128)")
except Exception as e:
    print(f"✗ Failed to load Custom CNN: {e}")

# Load MobileNetV2
try:
    models_to_evaluate['MobileNetV2'] = {
        'model': load_model(os.path.join(models_dir, 'mobilenetv2_raw_best.keras')),
        'img_size': 224,
        'preprocess': mobilenet_preprocess
    }
    print("✓ Loaded: MobileNetV2 (224x224)")
except Exception as e:
    print(f"✗ Failed to load MobileNetV2: {e}")

# Load EfficientNetB0
try:
    models_to_evaluate['EfficientNetB0'] = {
        'model': load_model(os.path.join(models_dir, 'efficientnetb0_raw_best.keras')),
        'img_size': 224,
        'preprocess': efficientnet_preprocess
    }
    print("✓ Loaded: EfficientNetB0 (224x224)")
except Exception as e:
    print(f"✗ Failed to load EfficientNetB0: {e}")

# Load ResNet50
try:
    models_to_evaluate['ResNet50'] = {
        'model': load_model(os.path.join(models_dir, 'resnet50_raw_best.keras')),
        'img_size': 224,
        'preprocess': resnet_preprocess
    }
    print("✓ Loaded: ResNet50 (224x224)")
except Exception as e:
    print(f"✗ Failed to load ResNet50: {e}")

print(f"\nTotal models loaded: {len(models_to_evaluate)}")

# ==============================================================================
# EVALUATE EACH MODEL
# ==============================================================================
print("\n" + "="*80)
print("EVALUATING MODELS ON TEST SET")
print("="*80)

results = {}

for model_name, model_info in models_to_evaluate.items():
    print(f"\nEvaluating {model_name}...")
    print("-" * 80)
    
    model = model_info['model']
    img_size = model_info['img_size']
    preprocess_fn = model_info['preprocess']
    
    # Create data generator with appropriate preprocessing
    if model_name == 'Custom CNN':
        # Custom CNN uses simple rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)
    else:
        # Transfer learning models use their specific preprocessing
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    
    test_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate
    print("Generating predictions...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    
    # Per-class accuracy
    class_accuracies = []
    for class_idx in range(NUM_CLASSES):
        class_mask = y_true == class_idx
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == y_true[class_mask]).mean()
            class_accuracies.append(class_acc)
    
    avg_per_class_acc = np.mean(class_accuracies)
    
    # Store results
    results[model_name] = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'avg_per_class_accuracy': avg_per_class_acc,
        'total_params': model.count_params(),
        'predictions': y_pred,
        'true_labels': y_true
    }
    
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Avg Per-Class Accuracy: {avg_per_class_acc*100:.2f}%")
    print(f"Total Parameters: {model.count_params():,}")

# ==============================================================================
# GENERATE COMPARISON TABLE
# ==============================================================================
print("\n" + "="*80)
print("GENERATING COMPARISON REPORT")
print("="*80)

# Create comparison DataFrame
comparison_data = []
for model_name, metrics in results.items():
    comparison_data.append({
        'Model': model_name,
        'Test Accuracy (%)': metrics['test_accuracy'] * 100,
        'Per-Class Accuracy (%)': metrics['avg_per_class_accuracy'] * 100,
        'Test Loss': metrics['test_loss'],
        'Parameters (M)': metrics['total_params'] / 1e6
    })

df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values('Test Accuracy (%)', ascending=False)

print("\n" + "="*80)
print("MODEL COMPARISON TABLE (RAW IMAGES)")
print("="*80)
print(df_comparison.to_string(index=False))

# Save comparison table
csv_path = os.path.join(models_dir, 'model_comparison_raw.csv')
df_comparison.to_csv(csv_path, index=False)
print(f"\n✓ Comparison table saved to: {csv_path}")

# Save detailed results
json_path = os.path.join(models_dir, 'evaluation_results_raw.json')
results_to_save = {
    model_name: {
        'test_accuracy': float(metrics['test_accuracy']),
        'test_loss': float(metrics['test_loss']),
        'avg_per_class_accuracy': float(metrics['avg_per_class_accuracy']),
        'total_params': int(metrics['total_params'])
    }
    for model_name, metrics in results.items()
}

with open(json_path, 'w') as f:
    json.dump(results_to_save, f, indent=4)
print(f"✓ Detailed results saved to: {json_path}")

# ==============================================================================
# VISUALIZATION: MODEL COMPARISON
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Comparison - RAW Images (No DIP Preprocessing)', fontsize=18, fontweight='bold')

# 1. Accuracy Comparison
ax1 = axes[0, 0]
models = df_comparison['Model'].values
accuracies = df_comparison['Test Accuracy (%)'].values
colors = ['#2ecc71' if acc > 90 else '#3498db' if acc > 85 else '#e74c3c' for acc in accuracies]
bars = ax1.barh(models, accuracies, color=colors, alpha=0.7)
ax1.set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax1.text(acc + 1, i, f'{acc:.2f}%', va='center', fontweight='bold')

# 2. Model Size Comparison
ax2 = axes[0, 1]
params = df_comparison['Parameters (M)'].values
bars = ax2.barh(models, params, color='#9b59b6', alpha=0.7)
ax2.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, (bar, p) in enumerate(zip(bars, params)):
    ax2.text(p + 0.5, i, f'{p:.2f}M', va='center', fontweight='bold')

# 3. Accuracy vs Model Size
ax3 = axes[1, 0]
ax3.scatter(params, accuracies, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
for i, model in enumerate(models):
    ax3.annotate(model, (params[i], accuracies[i]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
ax3.set_xlabel('Parameters (M)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('Efficiency: Accuracy vs Model Size', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Efficiency Score (Accuracy / Parameters)
ax4 = axes[1, 1]
efficiency = accuracies / params
bars = ax4.barh(models, efficiency, color='#e67e22', alpha=0.7)
ax4.set_xlabel('Efficiency Score (Accuracy/Param)', fontsize=12, fontweight='bold')
ax4.set_title('Model Efficiency Score', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for i, (bar, eff) in enumerate(zip(bars, efficiency)):
    ax4.text(eff + 0.5, i, f'{eff:.2f}', va='center', fontweight='bold')

plt.tight_layout()
comparison_plot_path = os.path.join(plots_dir, 'all_models_comparison_raw.png')
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Comparison plot saved to: {comparison_plot_path}")

# ==============================================================================
# CONFUSION MATRIX FOR BEST MODEL
# ==============================================================================
best_model_name = df_comparison.iloc[0]['Model']
best_model_results = results[best_model_name]

print(f"\nGenerating confusion matrix for best model: {best_model_name}")

cm = confusion_matrix(best_model_results['true_labels'], best_model_results['predictions'])

plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name} (Raw Images)', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

cm_plot_path = os.path.join(plots_dir, f'{best_model_name.lower().replace(" ", "_")}_confusion_matrix_raw.png')
plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved to: {cm_plot_path}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {df_comparison.iloc[0]['Test Accuracy (%)']:.2f}%")

print("\n" + "="*80)
print("COMPARISON: DIP vs RAW Images")
print("="*80)
print("\nWith DIP Preprocessing (128x128):")
print("  - Custom CNN: 67.88%")
print("  - MobileNetV2: 88.39%")
print("  - EfficientNetB0: 2.87%")
print("  - ResNet50: 35.49%")

print("\nWith RAW Images (224x224):")
for _, row in df_comparison.iterrows():
    print(f"  - {row['Model']}: {row['Test Accuracy (%)']:.2f}%")

print("\n✓ All models evaluated successfully!")
print(f"\nGenerated files:")
print(f"  - {csv_path}")
print(f"  - {json_path}")
print(f"  - {comparison_plot_path}")
print(f"  - {cm_plot_path}")

print("\n" + "="*80)
print("READY FOR PRESENTATION!")
print("="*80)
