import torch
from torchvision import transforms
from PIL import Image

# Image preprocessing for MobileNetV3
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(model_path):
    # Load MobileNetV3 Large
    model = torch.hub.load('pytorch/vision:v0.15.1', 'mobilenet_v3_large', pretrained=False)
    # Change classifier for 2 classes (Male/Female)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, image: Image.Image):
    img = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    return "Male" if pred.item() == 0 else "Female"
