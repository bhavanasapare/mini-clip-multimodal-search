import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from PIL import Image

# ---- Model Architecture (same as notebook) ----
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=256, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, projection_dim=256):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = ProjectionHead(embedding_dim=512, projection_dim=projection_dim)
    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return self.projection_head(features)

class TextEncoder(nn.Module):
    def __init__(self, projection_dim=256):
        super().__init__()
        self.transformer = AutoModel.from_pretrained("distilbert-base-uncased")
        self.projection_head = ProjectionHead(embedding_dim=768, projection_dim=projection_dim)
    def forward(self, input_ids, attention_mask):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]
        return self.projection_head(cls_token)

class MiniCLIP(nn.Module):
    def __init__(self, projection_dim=256, initial_temperature=0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(projection_dim=projection_dim)
        self.text_encoder = TextEncoder(projection_dim=projection_dim)
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())
    def forward(self, images, input_ids, attention_mask):
        image_embeds = self.image_encoder(images)
        text_embeds = self.text_encoder(input_ids, attention_mask)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        temperature = self.log_temperature.exp()
        logits = torch.matmul(image_embeds, text_embeds.t()) / temperature
        return logits

# ---- Load Model & Data (cached so it only runs once) ----
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = MiniCLIP()
    model.load_state_dict(torch.load("mini_clip_model.pth", map_location=device))
    model.eval()
    return model, device

@st.cache_resource
def load_gallery(_model, _device):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = load_dataset("jxie/flickr8k", split="test")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    gallery_images = []
    gallery_tensors = []
    for i in range(min(100, len(dataset))):
        img = dataset[i]["image"].convert("RGB")
        gallery_images.append(img)
        gallery_tensors.append(transform(img))
    
    gallery_tensor = torch.stack(gallery_tensors).to(_device)
    with torch.no_grad():
        gallery_embeds = _model.image_encoder(gallery_tensor)
        gallery_embeds = F.normalize(gallery_embeds, p=2, dim=-1)
    
    return gallery_images, gallery_embeds, tokenizer

# ---- Streamlit UI ----
st.set_page_config(page_title="Mini-CLIP Search", page_icon="🖼️", layout="wide")
st.title("🖼️ Mini-CLIP: Zero-Shot Image Search")
st.markdown("Custom PyTorch Dual-Tower Model trained with InfoNCE Loss on Flickr8k")

model, device = load_model()
gallery_images, gallery_embeds, tokenizer = load_gallery(model, device)

query = st.text_input("🔍 Enter your search query:", placeholder="e.g., a dog running outdoors")
num_results = st.slider("Number of results:", 1, 5, 3)

if query:
    tokens = tokenizer(query, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        text_emb = model.text_encoder(tokens["input_ids"], tokens["attention_mask"])
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        similarities = torch.matmul(text_emb, gallery_embeds.t()).squeeze(0)
        top_k = torch.topk(similarities, k=num_results)
    
    cols = st.columns(num_results)
    for i, (idx, score) in enumerate(zip(top_k.indices.numpy(), top_k.values.numpy())):
        with cols[i]:
            st.image(gallery_images[idx], caption=f"Score: {score:.3f}", use_container_width=True)
