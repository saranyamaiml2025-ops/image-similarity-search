import os
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Image Similarity Search", page_icon="🖼️", layout="wide")
st.title("🖼️ Image Similarity Search for a Small Dataset")
st.caption("A computer-vision capstone demo using colour histograms and cosine similarity.")

DATASET_DIR = Path(__file__).parent / "dataset"
SUPPORTED = {".png", ".jpg", ".jpeg", ".webp"}

def feature_vector(image, bins=8):
    """Return a normalized RGB colour-histogram feature vector."""
    image = ImageOps.exif_transpose(image).convert("RGB").resize((128, 128))
    arr = np.asarray(image)
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=bins, range=(0, 256))
        hist_parts.extend(hist.astype(float).tolist())
    vector = np.asarray(hist_parts, dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector

@st.cache_data
def load_dataset():
    records = []
    for path in sorted(DATASET_DIR.iterdir()):
        if path.suffix.lower() in SUPPORTED:
            try:
                with Image.open(path) as im:
                    records.append({"name": path.name, "path": str(path),
                                    "feature": feature_vector(im),
                                    "image": im.convert("RGB").copy()})
            except Exception:
                pass
    return records

records = load_dataset()
st.sidebar.header("Search settings")
top_k = st.sidebar.slider("Number of results", min_value=3, max_value=8, value=5)
st.sidebar.write(f"Dataset images: **{len(records)}**")
st.sidebar.info("Scores are similarity measures, not probabilities of correctness.")

uploaded = st.file_uploader("Upload a query image", type=["png", "jpg", "jpeg", "webp"])
if uploaded:
    query = Image.open(uploaded).convert("RGB")
    st.subheader("Query image")
    st.image(query, width=260)
    qvec = feature_vector(query)
    scored = []
    for rec in records:
        score = float(cosine_similarity([qvec], [rec["feature"]])[0][0])
        scored.append((score, rec))
    scored.sort(key=lambda item: item[0], reverse=True)
    st.subheader("Most visually similar dataset images")
    cols = st.columns(min(top_k, 4))
    for rank, (score, rec) in enumerate(scored[:top_k], start=1):
        with cols[(rank - 1) % len(cols)]:
            st.image(rec["image"], use_container_width=True)
            st.markdown(f"**Rank {rank}: {rec['name']}**")
            st.write(f"Similarity score: `{score:.3f}`")
    st.caption("This baseline compares colour distributions. It may rank unrelated images with similar colours highly.")
else:
    st.info("Upload any image to search the included demo dataset.")
    st.subheader("Included demo dataset")
    cols = st.columns(4)
    for i, rec in enumerate(records):
        with cols[i % 4]:
            st.image(rec["image"], use_container_width=True)
            st.caption(rec["name"])

st.divider()
st.subheader("Method and limitations")
st.markdown("""
- **Preprocessing:** resize images and convert to RGB.
- **Feature extraction:** concatenate 8-bin histograms for red, green and blue channels.
- **Similarity:** cosine similarity between normalized feature vectors.
- **Evaluation idea:** test query images, inspect top-ranked results, and record whether the results look relevant.
- **Limitation:** colour histograms do not understand object meaning, shape or scene context. Similar colours can produce high scores even for unrelated objects.
""")
