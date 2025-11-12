"""
Streamlit App for Image Classification
Loads the trained model and performs real-time image predictions
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import json
import io
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-bar {
        width: 100%;
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and class names"""
    try:
        model = keras.models.load_model('models/cifar10_classifier.h5')
        with open('models/class_names.json', 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except FileNotFoundError:
        st.error("Model files not found! Please run train_model.py first.")
        return None, None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to 32x32 (CIFAR-10 input size)
    img = image.resize((32, 32))
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Normalize to [0, 1]
    img_array = np.array(img, dtype='float32') / 255.0
    return img_array

# Title and description
st.title("🖼️ Image Classifier")
st.markdown("**AI-powered image classification using TensorFlow and CIFAR-10 trained model**")
st.divider()

# Load model
model, class_names = load_model()

if model is None or class_names is None:
    st.stop()

# Sidebar for information
with st.sidebar:
    st.header("About")
    st.info("""
    This app uses a Convolutional Neural Network (CNN) 
    trained on the CIFAR-10 dataset to classify images into 
    10 different categories.
    
    **Categories:**
    - Airplane
    - Automobile
    - Bird
    - Cat
    - Deer
    - Dog
    - Frog
    - Horse
    - Ship
    - Truck
    """)
    
    st.header("How to use")
    st.markdown("""
    1. Upload an image
    2. Click "Classify"
    3. View the predictions
    """)

# Main content tabs
tab1, tab2 = st.tabs(["Upload Image", "Sample Predictions"])

with tab1:
    st.subheader("Upload Your Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG, etc.)",
        type=['jpg', 'jpeg', 'png', 'gif', 'bmp']
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        with col2:
            st.subheader("Processing...")
            
            if st.button("🔍 Classify Image", key="classify_btn", use_container_width=True):
                # Preprocess and predict
                processed_img = preprocess_image(image)
                prediction = model.predict(np.array([processed_img]), verbose=0)
                predicted_class_idx = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class_idx]
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                # Main prediction
                st.metric(
                    label="Predicted Class",
                    value=class_names[predicted_class_idx].upper(),
                    delta=f"{confidence*100:.2f}% confidence"
                )
                
                # Show all predictions
                st.subheader("All Predictions")
                
                # Sort by confidence
                sorted_indices = np.argsort(prediction[0])[::-1]
                
                for idx in sorted_indices:
                    class_name = class_names[idx]
                    conf = prediction[0][idx]
                    
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.write(f"**{class_name}**")
                    with col2:
                        progress_bar_html = f"""
                        <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
                            <div style="width: {conf*100}%; background-color: #1f77b4; height: 20px;"></div>
                        </div>
                        <p style="text-align: right; margin: 0;">{conf*100:.1f}%</p>
                        """
                        st.markdown(progress_bar_html, unsafe_allow_html=True)

with tab2:
    st.subheader("Sample Predictions")
    st.info("Load sample images from CIFAR-10 test set to see how the model performs")
    
    if st.button("Load Sample Predictions", use_container_width=True):
        # Load CIFAR-10 test set
        (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0
        y_test = y_test.flatten()
        
        # Get 5 random samples
        indices = np.random.choice(len(x_test), 5, replace=False)
        
        for i, idx in enumerate(indices):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display image
                img_array = (x_test[idx] * 255).astype('uint8')
                st.image(img_array, caption=f"Sample {i+1}", use_column_width=True)
            
            with col2:
                # Make prediction
                prediction = model.predict(np.array([x_test[idx]]), verbose=0)
                predicted_idx = np.argmax(prediction[0])
                confidence = prediction[0][predicted_idx]
                true_class = class_names[y_test[idx]]
                predicted_class = class_names[predicted_idx]
                
                st.write(f"**True Class:** {true_class}")
                st.write(f"**Predicted:** {predicted_class}")
                st.write(f"**Confidence:** {confidence*100:.2f}%")
                
                # Show if prediction is correct
                if predicted_idx == y_test[idx]:
                    st.success("✅ Correct prediction!")
                else:
                    st.warning("❌ Incorrect prediction")
                
                st.divider()

st.divider()
st.markdown("""
---
**Created with ❤️ using TensorFlow, Keras, and Streamlit**
""")
