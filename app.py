import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

st.title("Gender Classification")

# ---------------------------
# Function to load model safely
# ---------------------------
@st.cache_resource
def load_model(model_path="model/mobilenet_v3_large_best.pth", num_classes=2):
    """
    Automatically detects if the .pth file is:
    - full model
    - state_dict
    - checkpoint with 'model_state_dict'
    """
    try:
        # Try loading full model directly
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        st.info("Loaded full model from .pth file.")
        return model
    except Exception:
        # If full model load fails, assume it's a state_dict or checkpoint
        model = models.mobilenet_v3_large(pretrained=False)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)

        checkpoint = torch.load(model_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            # Check if it's a checkpoint with 'model_state_dict'
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
              #  st.info("Loaded model_state_dict from checkpoint.")
            else:
                # Assume dict is state_dict
                model.load_state_dict(checkpoint)
              #  st.info("Loaded state_dict from .pth file.")
        else:
            raise ValueError("Unknown .pth file format!")

        model.eval()
        return model

# ---------------------------
# Load model
# ---------------------------
model = load_model()

# ---------------------------
# Image preprocessing
# ---------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ---------------------------
# Prediction
# ---------------------------
def predict(image):
    tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
    return "Male" if pred.item() == 0 else "Female"

# ---------------------------
# Streamlit UI
# ---------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    label = predict(image)
    st.success(f"Predicted Gender: {label}")
