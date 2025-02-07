import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

GDRIVE_FILE_ID = "1Xsxt5EsHLzsVhXolxDb23wOnybFcoZ-f"

def download_model():
    model_path = "trained_plant_disease_model.keras"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... Please wait."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", model_path, quiet=False)
    return model_path

def model_prediction(test_image):
    model_path = download_model()
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

from PIL import Image
img = Image.open('Diseases.png')
st.image(img)

if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", 
                unsafe_allow_html=True)

elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Detection System For Sustainable Agriculture')

test_image = st.file_uploader('Choose an image:')
if st.button('Show Image'):
    st.image(test_image, width=4, use_column_width=True)

if st.button('Predict'):
    st.snow()
    st.write('Our Prediction')
    result_index = model_prediction(test_image)
    class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    st.success(f'Model is predicting it’s a {class_name[result_index]}')
