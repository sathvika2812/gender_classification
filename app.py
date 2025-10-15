import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v3_large(pretrained=False)
    num_classes = 2  # Male/Female
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load("model/mobilenet_v3_large_best.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# ---------------------------
# Preprocessing
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
st.title("Gender Classification (MobileNetV3)")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    label = predict(image)
    st.success(f"Predicted Gender: {label}")
