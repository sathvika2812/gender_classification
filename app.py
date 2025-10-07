import streamlit as st
from PIL import Image
import cv2
import numpy as np
from utils import load_model, predict

st.set_page_config(page_title="Gender Classification", layout="centered")
st.title("Gender Classification App 🚻")

# Load model
model = load_model("model/mobilenet_v3_large_best.pth")

# Sidebar: Choose input type
input_type = st.sidebar.selectbox("Select Input Type", ["Upload Image", "Webcam"])

if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict Gender"):
            label = predict(model, image)
            st.success(f"Predicted Gender: {label}")

else:  # Webcam
    st.info("Webcam Input Selected. Press 'Capture' to take a photo.")
    cam = st.camera_input("Take a photo")
    if cam:
        image = Image.open(cam).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)
        if st.button("Predict Gender"):
            label = predict(model, image)
            st.success(f"Predicted Gender: {label}")
