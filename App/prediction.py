from urllib import response
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO

model = tf.keras.models.load_model("fire_detection_model.h5")


def predict_image(img):
    img_pred = np.array(img)
    img_pred = tf.image.resize(img_pred, size=[224, 224])
    img_pred = img_pred / 255.0
    res = int(tf.round(model.predict(x=tf.expand_dims(img_pred, axis=0))))
    res = "Fire Detected" if res == 0 else "No Fire Detected"
    st.title(res)
    st.image(img)


st.markdown(
    "<h1 style='text-align: center'> Forest Fire Detection </h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center'>This simple app is used to detect fire in the forest using a <strong>convolutional neural network</strong>.</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 4])
with col1:
    choose = st.radio("Select Image From", ("Upload", "URL"))

with col2:
    if choose == "Upload":
        file = st.file_uploader("Choose an image...", type=["jpg"])
        if file is not None:
            img = Image.open(file)
            button = st.button("Predict")
            if button:
                predict_image(img)
    else:
        url = st.text_area("Enter URL")
        if url:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                button = st.button("Predict")
                if button:
                    predict_image(img)
            except Exception as e:
                st.error("Invalid URL")
