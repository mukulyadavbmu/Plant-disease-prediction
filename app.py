"""
Plant Disease Detection - Interactive Streamlit App
Digital Image Processing Project
"""

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os
import json

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        margin-top: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #E8F5E9;
        margin: 10px 0;
    }
    .disease-name {
        font-size: 2rem;
        font-weight: bold;
        color: #1B5E20;
    }
    .confidence {
        font-size: 1.2rem;
        color: #388E3C;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🌿 Plant Disease Detection System</p>', unsafe_allow_html=True)
st.markdown("### Using Deep Learning & Digital Image Processing")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/plant.png", width=100)
    st.markdown("## About This Project")
    st.info(
        """
        This system uses:
        - **Deep Learning** (CNN)
        - **Advanced Image Processing**
        - **38 Plant Disease Classes**
        
        **DIP Techniques Applied:**
        - Bilateral Filtering
        - CLAHE Enhancement
        - Unsharp Masking
        """
    )
    
    st.markdown("## How to Use")
    st.markdown("""
    1. Upload a plant leaf image
    2. View the preprocessing steps
    3. Get instant disease prediction
    4. See confidence scores
    """)


@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = "baseline_cnn_model.keras"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model not found at {model_path}. Please train the model first!")
        return None


@st.cache_data
def load_class_names():
    """Load class names"""
    # These are the 38 classes from the Plant Diseases dataset
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


def advanced_preprocessing(image):
    """Apply DIP techniques to the image"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, (128, 128))
    
    steps = {}
    steps['original'] = img_resized.copy()
    
    # 1. Bilateral Filter for edge-preserving noise reduction
    bilateral = cv2.bilateralFilter(img_resized, d=9, sigmaColor=75, sigmaSpace=75)
    steps['bilateral'] = bilateral.copy()
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab_img = cv2.cvtColor(bilateral, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l_channel)
    clahe_lab_img = cv2.merge((clahe_l, a_channel, b_channel))
    clahe_rgb = cv2.cvtColor(clahe_lab_img, cv2.COLOR_LAB2RGB)
    steps['clahe'] = clahe_rgb.copy()
    
    # 3. Unsharp Masking for Sharpening
    gaussian_blur = cv2.GaussianBlur(clahe_rgb, (5, 5), 10.0)
    unsharp = cv2.addWeighted(clahe_rgb.astype('float32'), 1.5, 
                               gaussian_blur.astype('float32'), -0.5, 0)
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
    steps['final'] = unsharp.copy()
    
    # Normalize for model input
    processed = unsharp.astype('float32') / 255.0
    
    return processed, steps


def predict_disease(model, image, class_names):
    """Make prediction on the image"""
    processed_img, steps = advanced_preprocessing(image)
    
    # Add batch dimension
    img_batch = np.expand_dims(processed_img, axis=0)
    
    # Make prediction
    predictions = model.predict(img_batch, verbose=0)
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        {
            'disease': class_names[idx].replace('___', ' - ').replace('_', ' '),
            'confidence': float(predictions[0][idx] * 100)
        }
        for idx in top_3_idx
    ]
    
    return top_3_predictions, steps


# Load model and class names
model = load_model()
class_names = load_class_names()

# Main content
if model is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<p class="sub-header">📤 Upload Plant Leaf Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plant leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Predict button
            if st.button('🔍 Analyze Disease', type='primary', use_container_width=True):
                with st.spinner('Analyzing image... Please wait...'):
                    predictions, processing_steps = predict_disease(model, image, class_names)
                
                # Store in session state
                st.session_state.predictions = predictions
                st.session_state.processing_steps = processing_steps
    
    with col2:
        if 'predictions' in st.session_state:
            st.markdown('<p class="sub-header">🎯 Prediction Results</p>', unsafe_allow_html=True)
            
            predictions = st.session_state.predictions
            
            # Display top prediction
            st.markdown(f"""
                <div class="prediction-box">
                    <p class="disease-name">🌿 {predictions[0]['disease']}</p>
                    <p class="confidence">Confidence: {predictions[0]['confidence']:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display confidence chart
            st.markdown("#### Top 3 Predictions:")
            for i, pred in enumerate(predictions, 1):
                st.progress(pred['confidence'] / 100, 
                           text=f"{i}. {pred['disease']} - {pred['confidence']:.2f}%")
            
            # Health status
            if 'healthy' in predictions[0]['disease'].lower():
                st.success("✅ This plant appears to be HEALTHY!")
            else:
                st.warning("⚠️ Disease Detected! Consider treatment options.")
    
    # Show preprocessing steps
    if 'processing_steps' in st.session_state:
        st.markdown('<p class="sub-header">🔬 Image Processing Pipeline</p>', unsafe_allow_html=True)
        st.markdown("*Advanced DIP techniques applied to enhance detection accuracy*")
        
        steps = st.session_state.processing_steps
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image(steps['original'], caption='1. Original', use_container_width=True)
        with col2:
            st.image(steps['bilateral'], caption='2. Bilateral Filter', use_container_width=True)
        with col3:
            st.image(steps['clahe'], caption='3. CLAHE Enhancement', use_container_width=True)
        with col4:
            st.image(steps['final'], caption='4. Unsharp Masking', use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎓 Digital Image Processing Project | Plant Disease Detection using CNN</p>
        <p>Built with TensorFlow, OpenCV, and Streamlit</p>
    </div>
""", unsafe_allow_html=True)
