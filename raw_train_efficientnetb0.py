"""
Train EfficientNetB0 on RAW images (no DIP preprocessing)
Standard transfer learning approach with 224x224 images

Expected accuracy: 95-97% (likely best performer)
Expected time: 40-50 minutes
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kagglehub

print("="*80)
print("TRAINING EfficientNetB0 ON RAW IMAGES (224x224, No DIP)")
print("="*80)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 38

# Get dataset path
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
base_dir = os.path.join(dataset_path, 'New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)')
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

# Output directories
models_dir = 'models_raw'
plots_dir = 'output_plots_raw'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")

# ==============================================================================
# DATA GENERATORS
# ==============================================================================
print("\n" + "="*80)
print("CREATING DATA GENERATORS")
print("="*80)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # EfficientNet specific preprocessing
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"✓ Training samples: {train_generator.samples}")
print(f"✓ Validation samples: {valid_generator.samples}")
print(f"✓ Number of classes: {len(train_generator.class_indices)}")

# ==============================================================================
# BUILD MODEL
# ==============================================================================
print("\n" + "="*80)
print("BUILDING EfficientNetB0 MODEL")
print("="*80)

def build_efficientnetb0():
    """Build EfficientNetB0 with transfer learning"""
    base_model = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    return model, base_model

model, base_model = build_efficientnetb0()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Model built successfully!")
model.summary()

trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\nTotal parameters: {model.count_params():,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {model.count_params() - trainable_params:,}")

# ==============================================================================
# CALLBACKS
# ==============================================================================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(models_dir, 'efficientnetb0_raw_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ==============================================================================
# TRAINING
# ==============================================================================
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

start_time = time.time()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time/60:.1f}m {training_time%60:.0f}s")

# ==============================================================================
# SAVE MODEL AND HISTORY
# ==============================================================================
model_path = os.path.join(models_dir, 'efficientnetb0_raw.keras')
history_path = os.path.join(models_dir, 'efficientnetb0_raw_history.json')
plot_path = os.path.join(plots_dir, 'efficientnetb0_raw_training.png')

model.save(model_path)
print(f"\n✓ Model saved to: {model_path}")

history_dict = {
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']]
}

with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=4)
print(f"✓ Training history saved to: {history_path}")

# ==============================================================================
# PLOT TRAINING CURVES
# ==============================================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.title('EfficientNetB0 (Raw Images) - Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
plt.title('EfficientNetB0 (Raw Images) - Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Training plot saved to: {plot_path}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("EfficientNetB0 (RAW IMAGES) TRAINING COMPLETE!")
print("="*80)
final_train_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100
best_val_acc = max(history.history['val_accuracy']) * 100

print(f"Final Training Accuracy: {final_train_acc:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Training Time: {training_time/60:.0f}m {training_time%60:.0f}s")
print(f"\nModel saved: {model_path}")
print(f"History saved: {history_path}")
print(f"Plot saved: {plot_path}")
print("\nNext: Run raw_train_resnet50.py")
print("="*80)
