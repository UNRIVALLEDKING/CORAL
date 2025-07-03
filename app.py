import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import pathlib, json, time

# ------------------------------------------------------------------
# CONFIG -----------------------------------------------------------
# ------------------------------------------------------------------
MODEL_INFO = {
    "Marine (484 sp)": {
        "weights": "models/marine_model_retrained_weights.pth",
        "labels":  "labels/marine_labels.txt",
        "arch":    "resnet18",
        "input_sz": 224,
    },
    "Coral (28 sp)": {
        "weights": "models/coral_cnn.pth",
        "labels":  "labels/coral_labels.txt",
        "arch":    "resnet18",          # will still load custom head
        "input_sz": 224,
    },
}

@st.cache_resource(show_spinner=False)
def load_labels(path: str):
    txt = pathlib.Path(path).read_text().strip().splitlines()
    if not txt:
        st.error(f"Label file **{path}** is empty – fix or run `python fix_labels.py {path}`")
        st.stop()
    return txt

@st.cache_resource(show_spinner=False)
def load_model(item: str):
    info = MODEL_INFO[item]
    labels = load_labels(info["labels"])
    num_cls = len(labels)

    # backbone
    if info["arch"] == "resnet18":
        model = models.resnet18(weights=None)        # No Internet fetch
        model.fc = nn.Linear(model.fc.in_features, num_cls)
    else:
        raise ValueError("Unsupported arch", info["arch"])

    state = torch.load(info["weights"], map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, labels, info["input_sz"]

# ------------------------------------------------------------------
# UI ---------------------------------------------------------------
# ------------------------------------------------------------------
st.title("🐠 Unified Marine + Coral Species Classifier")
model_choice = st.sidebar.selectbox("Choose model", list(MODEL_INFO.keys()))

uploaded = st.file_uploader("Upload an underwater *jpg/png* or take a snapshot 👇",
                             type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input image", use_column_width=True)

    with st.spinner("Inferencing …"):
        model, classes, input_sz = load_model(model_choice)
        tfm = transforms.Compose([
            transforms.Resize((input_sz,input_sz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std =[0.229,0.224,0.225]),
        ])
        x = tfm(img).unsqueeze(0)            # (1,C,H,W)
        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=1).squeeze(0).numpy()

        # top-5 table
        topk = probs.argsort()[-5:][::-1]
        st.subheader("Top-5 predictions")
        for idx in topk:
            st.write(f"{classes[idx]:<30s} **{probs[idx]*100:5.1f}%**")

        # Grad-CAM
        target = [ClassifierOutputTarget(topk[0])]
        cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
        grayscale = cam(input_tensor=x, targets=target)[0]
        img_arr  = np.array(img.resize((input_sz,input_sz))) / 255.0
        cam_vis  = show_cam_on_image(img_arr, grayscale, use_rgb=True)
        st.image(cam_vis, caption="Grad-CAM heat-map", use_column_width=True)

