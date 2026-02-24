"""
Plant Disease Detection - Raw Image Models (No DIP)
Streamlit App for transfer learning models trained on raw 224x224 images
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection - Raw Models",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #e63946;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .model-card {
        background: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(246, 92, 139, 0.2);
        margin: 15px 0;
        border-left: 6px solid #f5576c;
    }
    .prediction-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 35px;
        border-radius: 25px;
        color: white;
        margin: 25px 0;
        box-shadow: 0 20px 50px rgba(250, 112, 154, 0.4);
        text-align: center;
    }
    .disease-name {
        font-size: 2.8rem;
        font-weight: 900;
        margin-bottom: 15px;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }
    .confidence-score {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 900;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 600;
    }
    .badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 700;
        margin: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .badge-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .badge-warning {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
    }
    .info-panel {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🚀 Plant Disease Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">⚡ High-Performance Transfer Learning Models (Raw 224×224 Images)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=100)
    st.markdown("## 🎯 Model Selection")
    
    model_choice = st.selectbox(
        "Choose Model:",
        ["MobileNetV2 (94.48%)", "EfficientNetB0 (Training...)", "ResNet50 (Pending)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("## 🏆 About These Models")
    st.success("""
    **Transfer Learning Approach:**
    - ✨ Pre-trained on ImageNet
    - ✨ 224×224 raw images (no DIP)
    - ✨ Model-specific preprocessing
    - ✨ Fine-tuned for plant diseases
    
    **Best Performer:** MobileNetV2 - 94.48%
    """)
    
    st.markdown("---")
    st.markdown("## 📈 Model Comparison")
    
    comparison_data = {
        'Model': ['MobileNetV2', 'EfficientNetB0', 'ResNet50'],
        'Status': ['✅ 94.48%', '⏳ Training', '⏳ Pending'],
        'Params': ['2.9M', '4.7M', '24.7M']
    }
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("## 🔥 Performance Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Best Accuracy", "94.48%", "↑ 6%")
    with col2:
        st.metric("Image Size", "224×224", "↑ 96px")

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_configs = {
        'MobileNetV2': {
            'path': 'models_raw/mobilenetv2_raw_best.keras',
            'preprocess': mobilenet_preprocess
        },
        'EfficientNetB0': {
            'path': 'models_raw/efficientnetb0_raw_best.keras',
            'preprocess': efficientnet_preprocess
        },
        'ResNet50': {
            'path': 'models_raw/resnet50_raw_best.keras',
            'preprocess': resnet_preprocess
        }
    }
    
    for name, config in model_configs.items():
        if os.path.exists(config['path']):
            try:
                models[name] = {
                    'model': tf.keras.models.load_model(config['path']),
                    'preprocess': config['preprocess']
                }
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

def preprocess_image(image, preprocess_fn):
    """Preprocess image for model"""
    img_array = np.array(image.resize((224, 224)))
    img_array = preprocess_fn(img_array)
    return img_array

def predict_disease(model_info, image, class_names):
    """Make prediction on the image"""
    model = model_info['model']
    preprocess_fn = model_info['preprocess']
    
    processed_img = preprocess_image(image, preprocess_fn)
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
    
    return top_5_predictions

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
        help="Upload a clear, high-resolution image of a plant leaf"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='📸 Uploaded Image', use_column_width=True)
        
        if st.button('🚀 Analyze with AI', type='primary', use_container_width=True):
            model_name = model_choice.split(' (')[0]
            model_info = models.get(model_name)
            
            if model_info is not None:
                with st.spinner('⚡ Processing with transfer learning...'):
                    predictions = predict_disease(model_info, image, class_names)
                
                st.session_state.predictions = predictions
                st.session_state.model_name = model_name
            else:
                st.warning(f"⏳ Model {model_name} is not ready yet!")

with col2:
    if 'predictions' in st.session_state:
        st.markdown("### 🎯 AI Prediction Results")
        
        predictions = st.session_state.predictions
        model_name = st.session_state.model_name
        
        # Top prediction
        top_pred = predictions[0]
        is_healthy = 'healthy' in top_pred['disease'].lower()
        
        st.markdown(f"""
            <div class="prediction-box">
                <div class="disease-name">{'🌟' if is_healthy else '⚠️'} {top_pred['disease']}</div>
                <div class="confidence-score">Confidence: {top_pred['confidence']:.2f}%</div>
                <span class="badge {'badge-success' if is_healthy else 'badge-warning'}">
                    {'HEALTHY PLANT' if is_healthy else 'DISEASE DETECTED'}
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        # Model info
        st.markdown(f"""
            <div class="info-panel">
                <strong>🤖 Model:</strong> {model_name}<br>
                <strong>📏 Input Size:</strong> 224×224 pixels<br>
                <strong>🎯 Approach:</strong> Transfer Learning (ImageNet)
            </div>
        """, unsafe_allow_html=True)
        
        # Top 5 predictions
        st.markdown("#### 📊 Confidence Distribution")
        
        pred_df = pd.DataFrame(predictions)
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=pred_df['disease'],
            x=pred_df['confidence'],
            orientation='h',
            marker=dict(
                color=pred_df['confidence'],
                colorscale='Plasma',
                line=dict(color='white', width=2)
            ),
            text=pred_df['confidence'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto',
            textfont=dict(size=14, color='white', family='Arial Black')
        ))
        
        fig.update_layout(
            xaxis_title='Confidence (%)',
            yaxis_title='',
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{top_pred['confidence']:.1f}%</div>
                    <div class="metric-label">Top Confidence</div>
                </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len([p for p in predictions if p['confidence'] > 10])}</div>
                    <div class="metric-label">Strong Candidates</div>
                </div>
            """, unsafe_allow_html=True)
        with col_m3:
            certainty = "High" if top_pred['confidence'] > 80 else "Medium" if top_pred['confidence'] > 50 else "Low"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{certainty}</div>
                    <div class="metric-label">Certainty Level</div>
                </div>
            """, unsafe_allow_html=True)

# Information section
if 'predictions' not in st.session_state:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="model-card">
                <h3>🎯 High Accuracy</h3>
                <p>Transfer learning models achieve 94%+ accuracy on plant disease classification</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="model-card">
                <h3>⚡ Fast Inference</h3>
                <p>Optimized architectures provide real-time predictions in milliseconds</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="model-card">
                <h3>🔬 38 Diseases</h3>
                <p>Trained on comprehensive dataset covering multiple crops and conditions</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 25px;'>
        <p style='font-size: 1.2rem; font-weight: 700;'>
            <span style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                🎓 Advanced Deep Learning Project
            </span>
        </p>
        <p style='color: #888; font-size: 1rem;'>
            Transfer Learning | TensorFlow | ImageNet Pre-training | 224×224 Resolution
        </p>
    </div>
""", unsafe_allow_html=True)
