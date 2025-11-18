import os
import torch
import clip
from PIL import Image

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Path to dataset
DATA_DIR = "data/train/images"

# Load all images
image_files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.endswith((".jpg", ".png"))
])

# Preprocess and encode all images
image_tensors = []
for img_file in image_files:
    img_path = os.path.join(DATA_DIR, img_file)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    image_tensors.append(image)

with torch.no_grad():
    image_features = torch.vstack([model.encode_image(img) for img in image_tensors])
    image_features /= image_features.norm(dim=-1, keepdim=True)  # normalize

# Example query text (from your text logs)
query_text = "Giraffe - Healthy, active"
text_tokens = clip.tokenize([query_text]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Compute cosine similarity
similarity = (image_features @ text_features.T).squeeze(1)  # [num_images]
best_idx = similarity.argmax().item()

print(f"Most similar image to '{query_text}': {image_files[best_idx]}")
