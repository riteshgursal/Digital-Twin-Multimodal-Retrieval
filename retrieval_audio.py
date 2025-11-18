import os
import torch
import clip
from PIL import Image
import torchaudio

# -------------------------------
# Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load CLIP model
# -------------------------------
model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------------------
# Audio embedding function
# -------------------------------
bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec_model = bundle.get_model().to(device).eval()

def audio_to_embedding(file_path):
    waveform, sample_rate = torchaudio.load(file_path)  # uses default backend
    # Resample if necessary
    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
        waveform = resampler(waveform)
    waveform = waveform.to(device)
    with torch.no_grad():
        emb = wav2vec_model(waveform).mean(dim=1)  # [1, feature_dim]
    emb = emb / emb.norm(dim=-1, keepdim=True)     # normalize
    return emb.squeeze(0)

# -------------------------------
# Dataset paths
# -------------------------------
IMG_DIR = "data/train/images"
TXT_DIR = "data/train/text_logs"
AUDIO_DIR = "data/train/audio"

# -------------------------------
# Load images
# -------------------------------
image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png"))])
image_tensors = [preprocess(Image.open(os.path.join(IMG_DIR, f))).unsqueeze(0).to(device) for f in image_files]

with torch.no_grad():
    image_features = torch.vstack([model.encode_image(img) for img in image_tensors])
    image_features /= image_features.norm(dim=-1, keepdim=True)

# -------------------------------
# Load texts
# -------------------------------
texts = []
for img_file in image_files:
    txt_file = os.path.splitext(img_file)[0] + ".txt"
    txt_path = os.path.join(TXT_DIR, txt_file)
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            texts.append(f.read().strip())
    else:
        texts.append("")  # Empty text if missing

text_tokens = clip.tokenize(texts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# -------------------------------
# Load audio embeddings (skip if missing)
# -------------------------------
audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])
audio_features = []
for f in audio_files:
    try:
        audio_features.append(audio_to_embedding(os.path.join(AUDIO_DIR, f)))
    except Exception as e:
        print(f"Warning: could not load {f}: {e}")
        audio_features.append(torch.zeros(bundle.get_model().encoder_embed_dim).to(device))  # fallback

audio_features = torch.vstack(audio_features)

# -------------------------------
# Example query
# -------------------------------
query_text = "Lion - Healthy, active"
query_audio_path = "data/train/audio/lion.wav"

with torch.no_grad():
    query_text_features = model.encode_text(clip.tokenize([query_text]).to(device))
    query_text_features /= query_text_features.norm(dim=-1, keepdim=True)

    if os.path.exists(query_audio_path):
        query_audio_features = audio_to_embedding(query_audio_path)
    else:
        query_audio_features = torch.zeros(bundle.get_model().encoder_embed_dim).to(device)

# -------------------------------
# Compute similarities
# -------------------------------
img_text_sim = (image_features @ query_text_features.T).squeeze(1)  # image-text similarity
audio_sim = (audio_features @ query_audio_features.T).squeeze(1)     # audio similarity

# Combine scores (weighted)
combined_sim = 0.5 * img_text_sim + 0.5 * audio_sim
best_idx = combined_sim.argmax().item()

print(f"Most similar item to query (text + audio): {image_files[best_idx]}")
