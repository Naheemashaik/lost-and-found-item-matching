import streamlit as st
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="Lost and Found Matching", page_icon="🔍", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    yolo = YOLO(BASE + "/best.pt")
    network = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    network = torch.nn.Sequential(*list(network.children())[:-1])
    network.eval()
    return yolo, network

model, resnet = load_models()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

features = np.load(BASE + "/Found_Object_Features.npy")

found_images = sorted([
    os.path.join(BASE, "Found_Items", f)
    for f in os.listdir(BASE + "/Found_Items")
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_object_features(image):
    result = model.predict(image, conf=0.15, verbose=False)[0]

    if len(result.boxes) > 0:
        best_box = result.boxes.conf.argmax()
        x1, y1, x2, y2 = result.boxes.xyxy[best_box].cpu().numpy()
        image = image.crop((int(x1), int(y1), int(x2), int(y2)))

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        vector = resnet(tensor)

    vector = vector.flatten().cpu().numpy()
    vector = vector / (np.linalg.norm(vector) + 1e-10)
    return vector

st.title("🔍 Lost and Found Item Matching System")
st.write("Upload your lost item image to find similar found items.")

uploaded_file = st.file_uploader(
    "Upload Lost Item Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    lost_image = Image.open(uploaded_file).convert("RGB")
    st.image(lost_image, caption="Uploaded Lost Item", width=300)

    if st.button("Find Matching Items"):
        with st.spinner("Finding similar items..."):
            lost_feature = extract_object_features(lost_image)
            scores = cosine_similarity(
                lost_feature.reshape(1, -1), features
            )[0]
            ranked = np.argsort(scores)[::-1]

        st.subheader("Matching Results")

        for index in ranked[:3]:
            st.image(found_images[index], width=250)
            st.write("Similarity Score:", f"{scores[index] * 100:.2f}%")
            st.divider()
