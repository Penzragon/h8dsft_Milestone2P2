from urllib import response
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO


def app():
    # Load the model
    model = tf.keras.models.load_model("fire_detection_model.h5")

    # Initialize img variable
    img = None

    # Create prediction function
    def predict_image(img):
        img_pred = np.array(img)
        img_pred = tf.image.resize(img_pred, size=[224, 224])
        img_pred = img_pred / 255.0
        res = int(tf.round(model.predict(x=tf.expand_dims(img_pred, axis=0))))
        res = "🚒 Fire Detected 🚒" if res == 0 else "✨ No Fire Detected ✨"
        title = f"<h2 style='text-align:center'>{res}</h2>"
        st.markdown(title, unsafe_allow_html=True)
        st.image(img, use_column_width=True)

    # Header Section
    st.markdown(
        "<h1 style='text-align: center'>🔥🌳 Wildfire Detection 🌳🔥</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center'>This simple app is used to detect wildfire using a <strong>convolutional neural network</strong></p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # Image Upload Option Section
    choose = st.selectbox("Choose an option", ["Upload Image", "From URL"])

    if choose == "Upload Image":  # If user chooses to upload image
        file = st.file_uploader("Choose an image...", type=["jpg"])
        if file is not None:
            img = Image.open(file)
    else:  # If user chooses to upload image from url
        url = st.text_area("Enter URL", placeholder="Paste the image URL here...")
        if url:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
            except:
                st.error(
                    "Failed to load the image. Please use a different URL or upload an image."
                )

    if img is not None:
        col1, col2, col3 = st.columns([1.7, 1, 1])
        with col2:
            predict = st.button("Predict 🧠")
        if predict:
            st.markdown("<hr>", unsafe_allow_html=True)
            col4, col5, col6 = st.columns([1, 3, 1])
            with col5:
                predict_image(img)
