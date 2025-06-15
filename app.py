# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Constants
IMAGE_SIZE_CORAL = 128
IMAGE_SIZE_MARINE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example class descriptions (can be expanded)
coral_class_descriptions = {
    "Acropora Cervicornis": "Staghorn coral, fast-growing, important reef builder.",
    "Acropora Palmata": "Elkhorn coral, large branching, Caribbean reefs.",
}
marine_class_descriptions = {
    "A73EGS-P": "Example marine species description.",
}

# CoralCNN with named last conv layer for Grad-CAM
class CoralCNN(nn.Module):
    def __init__(self, num_classes):
        super(CoralCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.flat_features = 128 * (IMAGE_SIZE_CORAL // 8) * (IMAGE_SIZE_CORAL // 8)
        self.fc = nn.Sequential(
            nn.Linear(self.flat_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.relu(self.bn32(self.conv1(x))))
        x = self.pool(self.relu(self.bn64(self.conv2(x))))
        x = self.pool(self.relu(self.bn128(self.final_conv(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def load_marine_model():
    with open("labels/marine_labels.txt") as f:
        classes = [line.strip() for line in f]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load("models/marine_model_retrained_weights.pth", map_location=DEVICE))
    model.eval()
    return model, classes

def load_coral_model():
    with open("labels/coral_labels.txt") as f:
        classes = [line.strip() for line in f]
    model = CoralCNN(len(classes)).to(DEVICE)
    model.load_state_dict(torch.load("models/corel_cnn.pth", map_location=DEVICE))
    model.eval()
    return model, classes

def predict_image(image, model, classes, image_size, normalize=True):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else transforms.Lambda(lambda x: x)
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top3_prob, top3_idx = torch.topk(probs, min(3, len(classes)))
        predictions = [(classes[i], float(prob)) for prob, i in zip(top3_prob[0], top3_idx[0])]
    return predictions, image

def get_gradcam(model, input_tensor, pred_idx, model_type):
    target_layers = [model.layer4] if model_type == "Marine Species" else [model.final_conv]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
    return grayscale_cam

# --- UI ---
st.sidebar.title("ℹ️ Model Info & Metadata")
model_type = st.selectbox("Choose Model", ["Marine Species", "Coral Species"])

if model_type == "Marine Species":
    st.sidebar.subheader("Marine Model")
    st.sidebar.write("**Architecture:** ResNet18")
    st.sidebar.write("**Input size:** 224x224")
    st.sidebar.write("**Classes:** 484")
    with st.sidebar.expander("Descriptions"):
        for k, v in marine_class_descriptions.items():
            st.write(f"**{k}**: {v}")
else:
    st.sidebar.subheader("Coral Model")
    st.sidebar.write("**Architecture:** CoralCNN")
    st.sidebar.write("**Input size:** 128x128")
    st.sidebar.write("**Classes:** 28")
    with st.sidebar.expander("Descriptions"):
        for k, v in coral_class_descriptions.items():
            st.write(f"**{k}**: {v}")

st.title("🌊 Unified Marine & Coral Species Classifier")
image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model_type == "Marine Species":
        model, classes = load_marine_model()
        image_size = IMAGE_SIZE_MARINE
        normalize = False
    else:
        model, classes = load_coral_model()
        image_size = IMAGE_SIZE_CORAL
        normalize = True

    predictions, input_tensor = predict_image(image, model, classes, image_size, normalize)
    pred_idx = classes.index(predictions[0][0])
    st.info(f"Model is trained on {len(classes)} classes.")

    tab1, tab2 = st.tabs(["🔍 Predictions", "🔥 Grad-CAM"])

    with tab1:
        st.subheader("Top Predictions")
        for i, (label, prob) in enumerate(predictions, 1):
            st.write(f"**{i}. {label}** - Confidence: {prob:.2%}")
            st.progress(prob)

    with tab2:
        st.subheader("Grad-CAM Visualization")
        np_img = np.array(image.resize((image_size, image_size))) / 255.0
        np_img = np.float32(np_img)
        grayscale_cam = get_gradcam(model, input_tensor, pred_idx, model_type)
        cam_image = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)
        st.image(cam_image, caption="Grad-CAM", use_column_width=True)
else:
    st.info("👆 Upload a JPG or PNG image.")
