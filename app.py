import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive model file ID
FILE_ID = "1Xsxt5EsHLzsVhXolxDb23wOnybFcoZ-f"
MODEL_PATH = "trained_plant_disease_model.keras"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if not available locally
if not os.path.exists(MODEL_PATH):
    st.warning("Downloading model from Google Drive... Please wait.")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Function to make predictions
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("🌿 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("📌 Select Page", ["Home", "Disease Recognition"])

# Displaying a header image
img = Image.open("Diseases.png")
st.image(img, use_column_width=True)

# Home Page
if app_mode == "Home":
    st.markdown("<h1 style='text-align: center;'>🌱 Plant Disease Detection System for Sustainable Agriculture 🌱</h1>", unsafe_allow_html=True)
    st.write("""
    This system helps in identifying plant diseases using deep learning. 
    Upload an image of a plant leaf to detect diseases and take preventive measures for sustainable agriculture. 
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("🔍 Plant Disease Detection")

    # Upload image
    test_image = st.file_uploader("📸 Upload a leaf image:", type=["jpg", "png", "jpeg"])
    
    # Show the uploaded image
    if test_image and st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    # Predict button
    if test_image and st.button("Predict"):
        st.snow()
        st.write("🔍 **Analyzing Image...**")
        
        result_index = model_prediction(test_image)
        class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
        
        st.success(f"✅ Model Prediction: **{class_names[result_index]}**")
