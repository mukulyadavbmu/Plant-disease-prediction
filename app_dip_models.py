"""
Plant Disease Detection - DIP Preprocessed Models
Streamlit App for models trained with DIP preprocessing (Bilateral, CLAHE, Unsharp Masking)
"""

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection - DIP Models",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    .main {
        background: #1a202c;
    }
    .stApp {
        background: #2d3748;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .model-card {
        background: #4a5568;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 10px 0;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease;
        color: #e2e8f0;
    }
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        text-align: center;
    }
    .disease-name {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .confidence-score {
        font-size: 1.8rem;
        font-weight: 600;
        opacity: 0.95;
    }
    .metric-container {
        background: #4a5568;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 1rem;
        color: #cbd5e0;
        margin-top: 5px;
        font-weight: 600;
    }
    .processing-step {
        background: #4a5568;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin: 10px 0;
        border-top: 3px solid #667eea;
    }
    .step-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #90cdf4;
        margin-bottom: 10px;
    }
    .info-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 5px;
    }
    .badge-healthy {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #155724;
    }
    .badge-diseased {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🌿 Plant Disease Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🔬 Advanced DIP Preprocessing Models (Bilateral + CLAHE + Unsharp Masking)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/microscope.png", width=100)
    st.markdown("## 🎯 Model Selection")
    
    model_choice = st.selectbox(
        "Choose Model:",
        ["Custom CNN (67.88%)", "MobileNetV2 (88.39%)", "ResNet50 (35.49%)", "EfficientNetB0 (2.87%)"],
        index=1
    )
    
    st.markdown("---")
    st.markdown("## 📊 About These Models")
    st.info("""
    **DIP Preprocessing Applied:**
    - 🔹 Bilateral Filter (noise reduction)
    - 🔹 CLAHE (contrast enhancement)
    - 🔹 Unsharp Masking (sharpening)
    
    **Image Size:** 128×128 (DIP models) or 224×224 (others)
    
    **Best Model:** MobileNetV2 (88.39%)
    """)
    
    st.markdown("---")
    st.markdown("## 📈 Model Comparison")
    
    comparison_data = {
        'Model': ['MobileNetV2', 'Custom CNN', 'ResNet50', 'EfficientNetB0'],
        'Accuracy': [88.39, 67.88, 35.49, 2.87]
    }
    df_comp = pd.DataFrame(comparison_data)
    
    fig = px.bar(df_comp, x='Model', y='Accuracy', 
                 color='Accuracy',
                 color_continuous_scale='Viridis',
                 labels={'Accuracy': 'Accuracy (%)'},
                 title='Model Performance')
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_paths = {
        'Custom CNN': 'baseline_cnn_model.keras',
        'MobileNetV2': 'models/mobilenetv2_best.keras',
        'EfficientNetB0': 'models/efficientnetb0_best.keras',
        'ResNet50': 'models/resnet50_best.keras'
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(path)
            except:
                models[name] = None
        else:
            models[name] = None
    
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

def advanced_preprocessing(image, target_size=128):
    """Apply DIP techniques to the image"""
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (target_size, target_size))
    
    steps = {}
    steps['original'] = img_resized.copy()
    
    # 1. Bilateral Filter
    bilateral = cv2.bilateralFilter(img_resized, d=9, sigmaColor=75, sigmaSpace=75)
    steps['bilateral'] = bilateral.copy()
    
    # 2. CLAHE
    lab_img = cv2.cvtColor(bilateral, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l_channel)
    clahe_lab_img = cv2.merge((clahe_l, a_channel, b_channel))
    clahe_rgb = cv2.cvtColor(clahe_lab_img, cv2.COLOR_LAB2RGB)
    steps['clahe'] = clahe_rgb.copy()
    
    # 3. Unsharp Masking
    gaussian_blur = cv2.GaussianBlur(clahe_rgb, (0, 0), 2.0)
    unsharp = cv2.addWeighted(clahe_rgb.astype('float32'), 1.5, 
                               gaussian_blur.astype('float32'), -0.5, 0)
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
    steps['final'] = unsharp.copy()
    
    processed = unsharp.astype('float32') / 255.0
    
    return processed, steps

def predict_disease(model, image, class_names, target_size=128):
    """Make prediction on the image"""
    processed_img, steps = advanced_preprocessing(image, target_size)
    img_batch = np.expand_dims(processed_img, axis=0)
    
    predictions = model.predict(img_batch, verbose=0)
    
    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    top_5_predictions = [
        {
            'disease': class_names[idx].replace('___', ' - ').replace('_', ' '),
            'confidence': float(predictions[0][idx] * 100)
        }
        for idx in top_5_idx
    ]
    
    return top_5_predictions, steps

# Load models
models = load_models()
class_names = load_class_names()

# Main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Plant Leaf Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='📸 Uploaded Image', use_column_width=True)
        
        # Determine target size based on model
        if 'Custom CNN' in model_choice:
            target_size = 128
        else:
            target_size = 128  # DIP models use 128
        
        if st.button('🔍 Analyze Disease', type='primary', use_container_width=True):
            model_name = model_choice.split(' (')[0]
            model = models.get(model_name)
            
            if model is not None:
                with st.spinner('🔬 Analyzing with DIP preprocessing...'):
                    predictions, processing_steps = predict_disease(model, image, class_names, target_size)
                
                st.session_state.predictions = predictions
                st.session_state.processing_steps = processing_steps
                st.session_state.model_name = model_name
            else:
                st.error(f"❌ Model {model_name} not found!")

with col2:
    if 'predictions' in st.session_state:
        st.markdown("### 🎯 Prediction Results")
        
        predictions = st.session_state.predictions
        model_name = st.session_state.model_name
        
        # Top prediction with beautiful styling
        top_pred = predictions[0]
        is_healthy = 'healthy' in top_pred['disease'].lower()
        
        st.markdown(f"""
            <div class="prediction-box">
                <div class="disease-name">{'🌟' if is_healthy else '⚠️'} {top_pred['disease']}</div>
                <div class="confidence-score">Confidence: {top_pred['confidence']:.2f}%</div>
                <span class="info-badge {'badge-healthy' if is_healthy else 'badge-diseased'}">
                    {'HEALTHY' if is_healthy else 'DISEASE DETECTED'}
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        # Model info
        st.markdown(f"**Model Used:** {model_name}")
        
        # Top 5 predictions with Plotly
        st.markdown("#### 📊 Top 5 Predictions")
        
        pred_df = pd.DataFrame(predictions)
        fig = go.Figure(data=[
            go.Bar(
                y=pred_df['disease'],
                x=pred_df['confidence'],
                orientation='h',
                marker=dict(
                    color=pred_df['confidence'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=pred_df['confidence'].apply(lambda x: f'{x:.2f}%'),
                textposition='auto',
            )
        ])
        fig.update_layout(
            xaxis_title='Confidence (%)',
            yaxis_title='',
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# Show preprocessing steps
if 'processing_steps' in st.session_state:
    st.markdown("---")
    st.markdown("### 🔬 DIP Preprocessing Pipeline")
    st.markdown("*Advanced digital image processing techniques enhance detection accuracy*")
    
    steps = st.session_state.processing_steps
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="processing-step">', unsafe_allow_html=True)
        st.markdown('<p class="step-title">1️⃣ Original</p>', unsafe_allow_html=True)
        st.image(steps['original'], use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="processing-step">', unsafe_allow_html=True)
        st.markdown('<p class="step-title">2️⃣ Bilateral Filter</p>', unsafe_allow_html=True)
        st.image(steps['bilateral'], use_column_width=True)
        st.markdown('<small>Noise reduction</small>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="processing-step">', unsafe_allow_html=True)
        st.markdown('<p class="step-title">3️⃣ CLAHE</p>', unsafe_allow_html=True)
        st.image(steps['clahe'], use_column_width=True)
        st.markdown('<small>Contrast enhancement</small>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="processing-step">', unsafe_allow_html=True)
        st.markdown('<p class="step-title">4️⃣ Unsharp Mask</p>', unsafe_allow_html=True)
        st.image(steps['final'], use_column_width=True)
        st.markdown('<small>Edge sharpening</small>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p style='font-size: 1.1rem; color: #667eea; font-weight: 600;'>
            🎓 Digital Image Processing Project | Plant Disease Detection
        </p>
        <p style='color: #888;'>
            Built with TensorFlow, OpenCV, and Streamlit | Models trained with DIP preprocessing
        </p>
    </div>
""", unsafe_allow_html=True)
