# Multi-Model Training Script for Plant Disease Detection
# This script trains 3 pre-trained models: MobileNetV2, EfficientNetB0, and ResNet50
# Your original Custom CNN model (baseline_cnn_model.keras) remains safe and untouched

import os
os.makedirs('output_plots', exist_ok=True)
os.makedirs('models', exist_ok=True)  # Separate folder for new models

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Digital Image Processing
import cv2

print("="*80)
print("MULTI-MODEL TRAINING FOR PLANT DISEASE DETECTION")
print("="*80)
print(f"TensorFlow Version: {tf.__version__}")
print(f"OpenCV Version: {cv2.__version__}")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 38

# Dataset paths (using cached dataset)
import kagglehub
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
base_dir = os.path.join(dataset_path, 'New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)')
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

print(f"\nDataset loaded from: {dataset_path}")
print(f"Training directory: {train_dir}")
print(f"Validation directory: {valid_dir}")

# ==============================================================================
# ADVANCED DIP PREPROCESSING FUNCTION
# ==============================================================================
def advanced_preprocessing(image):
    """
    Apply three advanced DIP techniques:
    1. Bilateral Filter - Edge-preserving noise reduction
    2. CLAHE (LAB color space) - Contrast enhancement
    3. Unsharp Masking - Edge sharpening
    """
    # Convert to uint8 if needed
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    # Step 1: Bilateral Filter (denoise while preserving edges)
    bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Step 2: CLAHE in LAB color space
    lab = cv2.cvtColor(bilateral, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    clahe_result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    # Step 3: Unsharp Masking (sharpen edges)
    gaussian = cv2.GaussianBlur(clahe_result, (0, 0), 2.0)
    unsharp = cv2.addWeighted(clahe_result, 1.5, gaussian, -0.5, 0)
    
    # Normalize to [0, 1] for model input
    return unsharp.astype(np.float32) / 255.0

# ==============================================================================
# DATA GENERATORS WITH PREPROCESSING
# ==============================================================================
print("\n" + "="*80)
print("SETTING UP DATA GENERATORS WITH ADVANCED DIP PREPROCESSING")
print("="*80)

train_datagen = ImageDataGenerator(
    preprocessing_function=advanced_preprocessing,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=advanced_preprocessing
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nTraining images: {train_generator.samples}")
print(f"Validation images: {validation_generator.samples}")
print(f"Number of classes: {NUM_CLASSES}")

# ==============================================================================
# MODEL BUILDING FUNCTIONS
# ==============================================================================

def build_mobilenetv2(num_classes):
    """Build MobileNetV2 model with custom top layers"""
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def build_efficientnetb0(num_classes):
    """Build EfficientNetB0 model with custom top layers"""
    base_model = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def build_resnet50(num_classes):
    """Build ResNet50 model with custom top layers"""
    base_model = ResNet50(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================

def train_model(model, model_name, train_gen, val_gen, epochs):
    """Train a model and save results"""
    print("\n" + "="*80)
    print(f"TRAINING: {model_name}")
    print("="*80)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    # Record start time
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)
    
    print(f"\n{model_name} training completed in {minutes}m {seconds}s")
    
    # Save model
    model_path = f'models/{model_name.lower().replace(" ", "_")}.keras'
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save training history
    history_path = f'models/{model_name.lower().replace(" ", "_")}_history.json'
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'training_time_seconds': training_time
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    # Plot training history
    plot_training_history(history, model_name, training_time)
    
    return history, training_time

def plot_training_history(history, model_name, training_time):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Time: {int(training_time//60)}m {int(training_time%60)}s', 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    
    plot_path = f'output_plots/{model_name.lower().replace(" ", "_")}_training.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to: {plot_path}")

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING MULTI-MODEL TRAINING")
    print("="*80)
    print(f"Total epochs per model: {EPOCHS}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    results = {}
    total_start_time = time.time()
    
    # ==============================================================================
    # 1. TRAIN MOBILENETV2
    # ==============================================================================
    print("\n" + "="*80)
    print("MODEL 1/3: MobileNetV2")
    print("="*80)
    mobilenet_model = build_mobilenetv2(NUM_CLASSES)
    mobilenet_history, mobilenet_time = train_model(
        mobilenet_model, 
        "MobileNetV2", 
        train_generator, 
        validation_generator, 
        EPOCHS
    )
    results['MobileNetV2'] = {
        'final_train_acc': float(mobilenet_history.history['accuracy'][-1]),
        'final_val_acc': float(mobilenet_history.history['val_accuracy'][-1]),
        'training_time': mobilenet_time
    }
    
    # ==============================================================================
    # 2. TRAIN EFFICIENTNETB0
    # ==============================================================================
    print("\n" + "="*80)
    print("MODEL 2/3: EfficientNetB0")
    print("="*80)
    efficientnet_model = build_efficientnetb0(NUM_CLASSES)
    efficientnet_history, efficientnet_time = train_model(
        efficientnet_model, 
        "EfficientNetB0", 
        train_generator, 
        validation_generator, 
        EPOCHS
    )
    results['EfficientNetB0'] = {
        'final_train_acc': float(efficientnet_history.history['accuracy'][-1]),
        'final_val_acc': float(efficientnet_history.history['val_accuracy'][-1]),
        'training_time': efficientnet_time
    }
    
    # ==============================================================================
    # 3. TRAIN RESNET50
    # ==============================================================================
    print("\n" + "="*80)
    print("MODEL 3/3: ResNet50")
    print("="*80)
    resnet_model = build_resnet50(NUM_CLASSES)
    resnet_history, resnet_time = train_model(
        resnet_model, 
        "ResNet50", 
        train_generator, 
        validation_generator, 
        EPOCHS
    )
    results['ResNet50'] = {
        'final_train_acc': float(resnet_history.history['accuracy'][-1]),
        'final_val_acc': float(resnet_history.history['val_accuracy'][-1]),
        'training_time': resnet_time
    }
    
    # ==============================================================================
    # SUMMARY AND COMPARISON
    # ==============================================================================
    total_time = time.time() - total_start_time
    total_hours = int(total_time // 3600)
    total_minutes = int((total_time % 3600) // 60)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE - ALL MODELS")
    print("="*80)
    print(f"Total training time: {total_hours}h {total_minutes}m")
    
    # Create comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    df = pd.DataFrame(results).T
    df['training_time_min'] = df['training_time'] / 60
    df = df.round(4)
    print(df.to_string())
    
    # Save results
    results_path = 'models/training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {results_path}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    model_names = list(results.keys())
    val_accs = [results[m]['final_val_acc'] * 100 for m in model_names]
    train_times = [results[m]['training_time'] / 60 for m in model_names]
    
    # Validation accuracy comparison
    bars1 = ax1.bar(model_names, val_accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Model Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax1.set_ylim([85, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Training time comparison
    bars2 = ax2.bar(model_names, train_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Training Time (minutes)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}m', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    comparison_path = 'output_plots/model_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {comparison_path}")
    
    print("\n" + "="*80)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*80)
    print("\nYour original baseline_cnn_model.keras is safe and untouched.")
    print(f"New models saved in: models/")
    print(f"- mobilenetv2.keras")
    print(f"- efficientnetb0.keras")
    print(f"- resnet50.keras")
    print("\nNext steps:")
    print("1. Run the model evaluation script to get test accuracy for each model")
    print("2. Compare all 4 models (Custom CNN + these 3) in your report")
