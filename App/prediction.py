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
    res = "🚒 Fire Detected 🚒" if res == 0 else "✨ No Fire Detected ✨"
    title = f"<h2 style='text-align:center'>{res}</h2>"
    st.markdown(title, unsafe_allow_html=True)
    st.image(img, use_column_width=True)


st.markdown(
    "<h1 style='text-align: center'>🔥🌳 Forest Fire Detection 🌳🔥</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center'>This simple app is used to detect fire in the forest using a <strong>convolutional neural network</strong>.</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

choose = st.selectbox("Choose an option", ["Upload Image", "From URL"])

if choose == "Upload Image":
    file = st.file_uploader("Choose an image...", type=["jpg"])
    if file is not None:
        img = Image.open(file)
        button = st.button("Predict")
        if button:
            predict_image(img)
else:
    url = st.text_area("Enter URL", placeholder="Paste the image URL here...")
    if url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            button = st.button("Predict")
            if button:
                predict_image(img)
        except:
            st.error(
                "Failed to load the image. Please use a different URL or upload an image."
            )
