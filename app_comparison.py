"""
Plant Disease Detection - Ultimate Comparison App
Side-by-side comparison of DIP vs Raw preprocessing approaches
"""

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection - Ultimate Comparison",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Ultimate styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 20px 0;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        letter-spacing: -2px;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        text-align: center;
        color: #764ba2;
        font-weight: 600;
        margin-bottom: 30px;
    }
    
    .comparison-card {
        background: white;
        padding: 30px;
        border-radius: 25px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
        margin: 20px 0;
        transition: all 0.4s ease;
        border: 3px solid transparent;
    }
    
    .comparison-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 70px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    .approach-badge {
        display: inline-block;
        padding: 10px 25px;
        border-radius: 30px;
        font-size: 1.1rem;
        font-weight: 800;
        margin: 10px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .badge-dip {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    .badge-raw {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        box-shadow: 0 5px 20px rgba(240, 147, 251, 0.4);
    }
    
    .prediction-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 30px;
        color: white;
        margin: 25px 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-panel::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .disease-title {
        font-size: 3rem;
        font-weight: 900;
        margin: 15px 0;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .confidence-display {
        font-size: 2.2rem;
        font-weight: 700;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .stat-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: white;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .winner-badge {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000;
        padding: 15px 30px;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 900;
        display: inline-block;
        box-shadow: 0 10px 30px rgba(255, 215, 0, 0.5);
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 10px 30px rgba(255, 215, 0, 0.5); }
        50% { box-shadow: 0 15px 50px rgba(255, 215, 0, 0.8); }
    }
    
    .comparison-table {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    .processing-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin: 20px 0;
    }
    
    .process-step {
        background: white;
        padding: 15px;
        border-radius: 18px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 4px solid #667eea;
    }
    
    .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 1.2rem;
        margin: 0 auto 10px;
    }
    
    .vs-divider {
        text-align: center;
        font-size: 3rem;
        font-weight: 900;
        color: #764ba2;
        margin: 30px 0;
        text-shadow: 0 5px 15px rgba(118, 75, 162, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown('<h1 class="hero-title">⚖️ Ultimate Model Comparison</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">🔬 DIP Preprocessing vs 🚀 Raw Transfer Learning</p>', unsafe_allow_html=True)

# Sidebar with comprehensive comparison
with st.sidebar:
    st.markdown("## 🎯 Approach Comparison")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin: 15px 0;'>
        <h3 style='margin: 0;'>🔬 DIP Approach</h3>
        <p style='margin: 5px 0; opacity: 0.9;'>• Bilateral + CLAHE + Unsharp</p>
        <p style='margin: 5px 0; opacity: 0.9;'>• 128×128 images</p>
        <p style='margin: 5px 0; opacity: 0.9;'>• Best: 88.39% (MobileNetV2)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 15px; color: white; margin: 15px 0;'>
        <h3 style='margin: 0;'>🚀 Raw Approach</h3>
        <p style='margin: 5px 0; opacity: 0.9;'>• No DIP preprocessing</p>
        <p style='margin: 5px 0; opacity: 0.9;'>• 224×224 images</p>
        <p style='margin: 5px 0; opacity: 0.9;'>• Best: 94.48% (MobileNetV2)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model selection for each approach
    st.markdown("### Select Models to Compare")
    
    dip_model = st.selectbox(
        "🔬 DIP Model:",
        ["MobileNetV2 (88.39%)", "Custom CNN (67.88%)", "ResNet50 (35.49%)"]
    )
    
    raw_model = st.selectbox(
        "🚀 Raw Model:",
        ["MobileNetV2 (94.48%)", "EfficientNetB0 (Training)", "ResNet50 (Pending)"]
    )
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("DIP Best", "88.39%")
    with col2:
        st.metric("Raw Best", "94.48%", "+6.09%")

@st.cache_resource
def load_all_models():
    """Load models from both approaches"""
    models = {
        'dip': {},
        'raw': {}
    }
    
    # DIP models
    dip_paths = {
        'MobileNetV2': 'models/mobilenetv2_best.keras',
        'Custom CNN': 'baseline_cnn_model.keras',
        'ResNet50': 'models/resnet50_best.keras'
    }
    
    for name, path in dip_paths.items():
        if os.path.exists(path):
            try:
                models['dip'][name] = tf.keras.models.load_model(path)
            except:
                models['dip'][name] = None
    
    # Raw models
    raw_paths = {
        'MobileNetV2': 'models_raw/mobilenetv2_raw_best.keras'
    }
    
    for name, path in raw_paths.items():
        if os.path.exists(path):
            try:
                models['raw'][name] = tf.keras.models.load_model(path)
            except:
                models['raw'][name] = None
    
    return models

@st.cache_data
def load_class_names():
    """Load class names"""
    classes = [
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
    return sorted(classes)

def dip_preprocess(image):
    """DIP preprocessing pipeline"""
    img_array = np.array(image.resize((128, 128)))
    bilateral = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)
    lab_img = cv2.cvtColor(bilateral, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab_img = cv2.merge((l, a, b))
    clahe_rgb = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    gaussian_blur = cv2.GaussianBlur(clahe_rgb, (0, 0), 2.0)
    unsharp = cv2.addWeighted(clahe_rgb.astype('float32'), 1.5, 
                               gaussian_blur.astype('float32'), -0.5, 0)
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
    return unsharp.astype('float32') / 255.0

def raw_preprocess(image):
    """Raw preprocessing for transfer learning"""
    img_array = np.array(image.resize((224, 224)))
    return mobilenet_preprocess(img_array)

def predict_both(models, image, class_names):
    """Predict with both approaches"""
    results = {}
    
    # DIP prediction
    dip_img = dip_preprocess(image)
    dip_batch = np.expand_dims(dip_img, axis=0)
    dip_preds = models['dip']['MobileNetV2'].predict(dip_batch, verbose=0)
    dip_top = np.argmax(dip_preds[0])
    results['dip'] = {
        'disease': class_names[dip_top].replace('___', ' - ').replace('_', ' '),
        'confidence': float(dip_preds[0][dip_top] * 100)
    }
    
    # Raw prediction
    raw_img = raw_preprocess(image)
    raw_batch = np.expand_dims(raw_img, axis=0)
    raw_preds = models['raw']['MobileNetV2'].predict(raw_batch, verbose=0)
    raw_top = np.argmax(raw_preds[0])
    results['raw'] = {
        'disease': class_names[raw_top].replace('___', ' - ').replace('_', ' '),
        'confidence': float(raw_preds[0][raw_top] * 100)
    }
    
    return results

# Load models
models = load_all_models()
class_names = load_class_names()

# Main content
uploaded_file = st.file_uploader(
    "📤 Upload Plant Leaf Image for Side-by-Side Comparison",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a high-quality image to compare both approaches"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns([1, 0.1, 1])
    
    with col1:
        st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
        st.image(image, caption='📸 Original Image', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if st.button('⚖️ Compare Both Approaches', type='primary', use_container_width=True):
            if models['dip'].get('MobileNetV2') and models['raw'].get('MobileNetV2'):
                with st.spinner('🔄 Running both models...'):
                    results = predict_both(models, image, class_names)
                st.session_state.results = results
            else:
                st.error("Models not loaded!")
    
    if 'results' in st.session_state:
        results = st.session_state.results
        
        st.markdown('<div class="vs-divider">⚔️ VS ⚔️</div>', unsafe_allow_html=True)
        
        col_dip, col_vs, col_raw = st.columns([1, 0.1, 1])
        
        with col_dip:
            st.markdown('<span class="approach-badge badge-dip">🔬 DIP Approach</span>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="prediction-panel" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <div class="disease-title">{results['dip']['disease']}</div>
                    <div class="confidence-display">Confidence: {results['dip']['confidence']:.2f}%</div>
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-value">128</div>
                            <div class="stat-label">Image Size</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">DIP</div>
                            <div class="stat-label">Preprocessing</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_raw:
            st.markdown('<span class="approach-badge badge-raw">🚀 Raw Approach</span>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="prediction-panel" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <div class="disease-title">{results['raw']['disease']}</div>
                    <div class="confidence-display">Confidence: {results['raw']['confidence']:.2f}%</div>
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-value">224</div>
                            <div class="stat-label">Image Size</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">RAW</div>
                            <div class="stat-label">Preprocessing</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Winner announcement
        winner = "Raw" if results['raw']['confidence'] > results['dip']['confidence'] else "DIP"
        diff = abs(results['raw']['confidence'] - results['dip']['confidence'])
        
        st.markdown(f"""
            <div style='text-align: center; margin: 40px 0;'>
                <div class="winner-badge">
                    🏆 {winner} Approach Wins by {diff:.2f}%!
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Detailed comparison table
        st.markdown("### 📊 Detailed Comparison")
        comparison_df = pd.DataFrame({
            'Metric': ['Predicted Disease', 'Confidence', 'Image Size', 'Preprocessing', 'Model Accuracy'],
            'DIP Approach': [
                results['dip']['disease'],
                f"{results['dip']['confidence']:.2f}%",
                '128×128',
                'Bilateral + CLAHE + Unsharp',
                '88.39%'
            ],
            'Raw Approach': [
                results['raw']['disease'],
                f"{results['raw']['confidence']:.2f}%",
                '224×224',
                'ImageNet Standard',
                '94.48%'
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 30px;'>
        <p style='font-size: 1.5rem; font-weight: 900;'>
            <span style='background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                🎓 Comprehensive Deep Learning Comparison Study
            </span>
        </p>
        <p style='color: #888; font-size: 1rem; font-weight: 600;'>
            Digital Image Processing vs Transfer Learning | TensorFlow | Plant Disease Detection
        </p>
    </div>
""", unsafe_allow_html=True)
