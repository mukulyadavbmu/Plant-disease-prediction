"""
Clean Demo Script - Plant Disease Detection
This script demonstrates the trained model without all the training code
Perfect for showing to teachers or testing the model quickly
"""

import tensorflow as tf
import cv2
import numpy as np
import json
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt

print("="*80)
print("PLANT DISEASE DETECTION - DEMO SCRIPT")
print("="*80)

# ==============================================================================
# 1. LOAD THE TRAINED MODEL
# ==============================================================================
print("\n[1/4] Loading trained model...")
model_path = 'baseline_cnn_model.keras'

if not os.path.exists(model_path):
    print(f"❌ ERROR: Model file not found at '{model_path}'")
    print("Please run the training script first (dip_project_training.py)")
    exit()

model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Load class names
class_indices_path = 'class_indices.json'
if os.path.exists(class_indices_path):
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    class_names = sorted(class_indices.keys())
else:
    # Default class names if file not found
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

print(f"✅ Loaded {len(class_names)} disease classes")

# ==============================================================================
# 2. ADVANCED PREPROCESSING FUNCTION
# ==============================================================================
def advanced_preprocessing(image_path, show_steps=True):
    """
    Apply DIP preprocessing pipeline to an image
    
    Args:
        image_path: Path to the image file
        show_steps: Whether to save intermediate steps
    
    Returns:
        processed_image: Ready for model input
        steps: Dictionary of intermediate images (if show_steps=True)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (128, 128))
    
    steps = {'original': img_resized.copy()} if show_steps else {}
    
    # Step 1: Bilateral Filter (edge-preserving noise reduction)
    bilateral = cv2.bilateralFilter(img_resized, d=9, sigmaColor=75, sigmaSpace=75)
    if show_steps:
        steps['bilateral'] = bilateral.copy()
    
    # Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab_img = cv2.cvtColor(bilateral, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l_channel)
    clahe_lab = cv2.merge((clahe_l, a_channel, b_channel))
    clahe_rgb = cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2RGB)
    if show_steps:
        steps['clahe'] = clahe_rgb.copy()
    
    # Step 3: Unsharp Masking (sharpening)
    gaussian_blur = cv2.GaussianBlur(clahe_rgb, (5, 5), 10.0)
    unsharp = cv2.addWeighted(clahe_rgb.astype('float32'), 1.5, 
                               gaussian_blur.astype('float32'), -0.5, 0)
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
    if show_steps:
        steps['final'] = unsharp.copy()
    
    # Normalize for model
    processed = unsharp.astype('float32') / 255.0
    
    return processed, steps


def predict_disease(image_path, top_k=3):
    """
    Predict disease from image with preprocessing visualization
    
    Args:
        image_path: Path to the leaf image
        top_k: Number of top predictions to return
    
    Returns:
        predictions: List of (class_name, confidence) tuples
        steps: Preprocessing steps for visualization
    """
    print(f"\n[2/4] Preprocessing image: {os.path.basename(image_path)}")
    processed_img, steps = advanced_preprocessing(image_path, show_steps=True)
    
    print("[3/4] Running model prediction...")
    img_batch = np.expand_dims(processed_img, axis=0)
    predictions = model.predict(img_batch, verbose=0)
    
    # Get top K predictions
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    results = [
        (class_names[idx].replace('___', ' - ').replace('_', ' '), 
         float(predictions[0][idx] * 100))
        for idx in top_indices
    ]
    
    return results, steps


def visualize_prediction(image_path, predictions, steps, output_path='demo_result.png'):
    """Create a comprehensive visualization of the prediction"""
    fig = plt.figure(figsize=(16, 10))
    
    # Original image
    ax1 = plt.subplot(2, 4, 1)
    original = Image.open(image_path)
    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Preprocessing steps
    step_titles = [
        ('original', '1. Resized (128x128)'),
        ('bilateral', '2. Bilateral Filter'),
        ('clahe', '3. CLAHE Enhancement'),
        ('final', '4. Unsharp Masking')
    ]
    
    for idx, (key, title) in enumerate(step_titles, start=2):
        ax = plt.subplot(2, 4, idx)
        ax.imshow(steps[key])
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    # Prediction results
    ax_pred = plt.subplot(2, 2, 3)
    ax_pred.axis('off')
    
    # Top prediction (large)
    disease, conf = predictions[0]
    color = 'green' if 'healthy' in disease.lower() else 'red'
    result_text = f"🌿 PREDICTION\n\n"
    result_text += f"{disease}\n\n"
    result_text += f"Confidence: {conf:.1f}%"
    
    ax_pred.text(0.5, 0.7, result_text, 
                 ha='center', va='center', fontsize=16,
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
                 fontweight='bold')
    
    # Top 3 bar chart
    ax_bar = plt.subplot(2, 2, 4)
    diseases = [p[0][:30] + '...' if len(p[0]) > 30 else p[0] for p in predictions]
    confidences = [p[1] for p in predictions]
    colors_bar = ['green' if 'healthy' in d.lower() else 'orange' for d in diseases]
    
    ax_bar.barh(range(len(predictions)), confidences, color=colors_bar, alpha=0.7)
    ax_bar.set_yticks(range(len(predictions)))
    ax_bar.set_yticklabels(diseases, fontsize=10)
    ax_bar.set_xlabel('Confidence (%)', fontsize=12)
    ax_bar.set_title('Top 3 Predictions', fontsize=14, fontweight='bold')
    ax_bar.invert_yaxis()
    
    for i, v in enumerate(confidences):
        ax_bar.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
    
    plt.suptitle('Plant Disease Detection - Complete Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs('demo_results', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {output_path}")


# ==============================================================================
# 3. DEMO EXECUTION
# ==============================================================================
def run_demo(test_image_path):
    """Run a complete demo on a single image"""
    if not os.path.exists(test_image_path):
        print(f"❌ Image not found: {test_image_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYZING IMAGE: {os.path.basename(test_image_path)}")
    print(f"{'='*80}")
    
    # Make prediction
    predictions, steps = predict_disease(test_image_path)
    
    print("\n[4/4] Results:")
    print("-" * 80)
    for i, (disease, confidence) in enumerate(predictions, 1):
        status = "🟢" if 'healthy' in disease.lower() else "🔴"
        print(f"{i}. {status} {disease}")
        print(f"   Confidence: {confidence:.2f}%")
    print("-" * 80)
    
    # Create visualization
    output_name = f"demo_results/{os.path.splitext(os.path.basename(test_image_path))[0]}_result.png"
    visualize_prediction(test_image_path, predictions, steps, output_name)
    
    return predictions


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEMO OPTIONS:")
    print("="*80)
    print("1. Test with validation dataset images")
    print("2. Test with custom image")
    print("="*80)
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Test with dataset images
        dataset_cache = r"C:\Users\mukul\.cache\kagglehub\datasets\vipoooool\new-plant-diseases-dataset\versions\2"
        valid_dir = os.path.join(dataset_cache, 'New Plant Diseases Dataset(Augmented)', 
                                  'New Plant Diseases Dataset(Augmented)', 'valid')
        
        if os.path.exists(valid_dir):
            # Get random images from different classes
            classes = os.listdir(valid_dir)
            print(f"\nFound {len(classes)} classes in validation set")
            print("Testing 3 random images...\n")
            
            import random
            for i in range(3):
                random_class = random.choice(classes)
                class_dir = os.path.join(valid_dir, random_class)
                images = os.listdir(class_dir)
                random_image = random.choice(images)
                image_path = os.path.join(class_dir, random_image)
                
                run_demo(image_path)
                print("\n")
        else:
            print("❌ Validation dataset not found. Please run training first.")
    
    elif choice == "2":
        # Test with custom image
        image_path = input("\nEnter full path to image: ").strip().strip('"')
        if os.path.exists(image_path):
            run_demo(image_path)
        else:
            print(f"❌ File not found: {image_path}")
    
    else:
        print("Invalid choice!")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nCheck the 'demo_results' folder for visualizations!")
