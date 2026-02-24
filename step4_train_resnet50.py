"""
STEP 4: Train ResNet50 Model
Trains ResNet50 on preprocessed dataset.
Run after step3_train_efficientnetb0.py completes.

Expected time: ~50-60 minutes
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*80)
print("STEP 4: TRAINING ResNet50")
print("="*80)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 38

# Paths
preprocessed_train = 'preprocessed_data/train'
preprocessed_valid = 'preprocessed_data/valid'
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Check if preprocessed data exists
if not os.path.exists(preprocessed_train):
    print("\n" + "="*80)
    print("ERROR: Preprocessed data not found!")
    print("="*80)
    print("Please run step1_preprocess_dataset.py first.")
    exit(1)

print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")

# ==============================================================================
# DATA GENERATORS
# ==============================================================================
print("\n" + "="*80)
print("LOADING PREPROCESSED DATA")
print("="*80)

# Only need rescaling since images are already preprocessed
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    preprocessed_train,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    preprocessed_valid,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nTraining images: {train_generator.samples}")
print(f"Validation images: {valid_generator.samples}")
print(f"Number of classes: {NUM_CLASSES}")

# ==============================================================================
# BUILD MODEL
# ==============================================================================
print("\n" + "="*80)
print("BUILDING ResNet50 MODEL")
print("="*80)

base_model = ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================================================================
# CALLBACKS
# ==============================================================================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
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
        f'{models_dir}/resnet50_best.keras',
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
minutes = int(training_time // 60)
seconds = int(training_time % 60)

print(f"\nTraining completed in {minutes}m {seconds}s")

# ==============================================================================
# SAVE MODEL AND RESULTS
# ==============================================================================
model_path = f'{models_dir}/resnet50.keras'
model.save(model_path)
print(f"\nModel saved to: {model_path}")

# Save training history
history_dict = {
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'training_time_seconds': training_time,
    'final_train_accuracy': float(history.history['accuracy'][-1]),
    'final_val_accuracy': float(history.history['val_accuracy'][-1])
}

history_path = f'{models_dir}/resnet50_history.json'
with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=4)
print(f"Training history saved to: {history_path}")

# ==============================================================================
# PLOT TRAINING HISTORY
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_title('ResNet50 - Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('ResNet50 - Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle(f'Training Time: {minutes}m {seconds}s | Final Val Acc: {history_dict["final_val_accuracy"]*100:.2f}%', 
             fontsize=12, y=1.02)
plt.tight_layout()

plot_path = 'output_plots/resnet50_training.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Training plot saved to: {plot_path}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("ResNet50 TRAINING COMPLETE!")
print("="*80)
print(f"Final Training Accuracy: {history_dict['final_train_accuracy']*100:.2f}%")
print(f"Final Validation Accuracy: {history_dict['final_val_accuracy']*100:.2f}%")
print(f"Training Time: {minutes}m {seconds}s")
print(f"\nModel saved: {model_path}")
print(f"History saved: {history_path}")
print(f"Plot saved: {plot_path}")
print("\nNext step: Run step5_evaluate_all_models.py")
print("="*80)
