import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Configuration
MODEL_PATH = "/data-mount/sathvika/projects/gender_classification/model/mobilenet_v3_large_best.pth"
LABELS_PATH = "labels.txt"
IMG_SIZE = 224

# Load class labels
@st.cache_resource
def load_labels():
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Load model
@st.cache_resource
def load_model():
    # Initialize MobileNetV3 (adjust architecture to match your training)
    model = torch.hub.load(
        'pytorch/vision:v0.10.0', 
        'mobilenet_v3_large',  # or 'mobilenet_v3_small'
        pretrained=False
    )
    
    # Modify classifier to match your number of classes
    num_classes = len(load_labels())
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocessing pipeline
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(model, image, labels):
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    return labels[predicted_idx.item()], confidence.item()

# Streamlit app
def main():
    st.set_page_config(page_title="MobileNetV3 Classifier", layout="wide")
    st.title("🚀 MobileNetV3 Image Classifier")
    st.markdown("Upload an image to classify using your custom trained model")

    # Load resources
    try:
        labels = load_labels()
        model = load_model()
        st.success(f"✅ Model loaded! ({len(labels)} classes)")
    except Exception as e:
        st.error(f"❌ Error loading model/resources: {str(e)}")
        st.stop()

    # Input options
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
    
    with col2:
        st.subheader("🖼️ Preview")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_column_width=True)
        else:
            st.info("Upload an image to see preview")

    # Prediction
    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing image..."):
            # Preprocess
            input_tensor = preprocess_image(image)
            
            # Predict
            predicted_class, confidence = predict(model, input_tensor, labels)
            
            # Results
            st.subheader("📊 Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Class", predicted_class)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Show all probabilities
            with st.expander("Show detailed probabilities"):
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                prob_data = {
                    "Class": labels,
                    "Probability": [f"{p:.4f}" for p in probabilities.tolist()]
                }
                st.table(prob_data)

if __name__ == "__main__":
    main()