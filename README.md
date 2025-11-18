<h1>Digital Twin Multimodal Retrieval</h1>

This project demonstrates a multimodal retrieval system designed for digital twin applications, where images and textual data are aligned in a shared embedding space for efficient search and analysis. It leverages OpenAI CLIP for real-time image-text similarity and can be extended to integrate additional modalities such as sensor readings, audio, and video.

---

**Project Description**

Modern digital twin systems generate massive amounts of heterogeneous data—images, textual logs, sensor readings, and more. Multimodal analysis and retrieval helps in:

Integrating and understanding multiple data types.

Searching efficiently for relevant information across modalities.

Enhancing decision-making in real-world systems.

This repository presents a simple, ready-to-run demo using a small dataset (5 images + corresponding text logs) to simulate retrieval tasks. Users can query the system with a text description and retrieve the most relevant image in real-time.

---

**Features**

Multimodal Retrieval: Retrieve images based on textual queries using CLIP embeddings.

Digital Twin Ready: Can be extended to simulate asset monitoring and state retrieval in digital twin systems.

Lightweight & Runnable: Works with minimal dataset, easy to run and test.

Pre-trained Models: Uses CLIP (ViT-B/32) for powerful, pre-aligned text-image embeddings.

Extensible: Easily integrate new modalities (sensor data, video frames) for enhanced retrieval.

---

**Real-Time Applications**

Smart Manufacturing:

*Query textual maintenance logs to retrieve images of machines under similar operating conditions.*

*Quickly detect anomalies or find historical examples of equipment states.*

Smart Agriculture:

*Retrieve field images from textual observations (e.g., “corn crop, healthy, no pest damage”) to monitor crop health over time.*

Predictive Maintenance:

*Align images of equipment with sensor logs for real-time fault diagnosis.*

Digital Twin Systems:

*Each physical asset has a “digital twin” in the system.*

*Multimodal retrieval allows querying one modality (e.g., a sensor reading or textual description) to retrieve other asset states, improving monitoring and simulation.*

---

**Project Structure**

```
digital-twin-multimodal-retrieval/
 ├─ data/
 │   ├─ train/
 │   │   ├─ images/       # sample images (5 animals)
 │   │   ├─ text_logs/    # corresponding textual descriptions
 │   ├─ test/
 │   │   ├─ images/       # optional test images
 │   │   ├─ text_logs/    # optional test text files
 ├─ src/
 │   ├─ dataset.py         # Dataset loader for images + text
 │   ├─ retrieval.py       # CLIP-based retrieval script
 ├─ requirements.txt       # Python dependencies
 ├─ README.md
 └─ .gitignore
```

---

**Setup Instructions**

Clone the repository:
```
git clone https://github.com/<riteshgursal>/digital-twin-multimodal-retrieval.git
cd digital-twin-multimodal-retrieval
```

Install dependencies:
```
pip install -r requirements.txt
```

Add your dataset:
```
Place images in data/train/images and data/test/images.
```
```
Place corresponding text logs in data/train/text_logs and data/test/text_logs.
```

Ensure filenames match (e.g., lion.jpg ↔ lion.txt).

Running the Demo
```
python src/retrieval.py
```

**Expected Output:**

Most similar image to 'Penguin - Healthy, active': penguin.jpg


The system compares the query text with all images in the dataset and retrieves the most relevant match using CLIP embeddings.

---

**Acknowledgements**

OpenAI CLIP
- for pretrained multimodal embeddings.

- PyTorch and Torchvision for deep learning utilities.

  ---
