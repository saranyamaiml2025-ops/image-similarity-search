
from pathlib import Path
import csv

import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Folder containing the demo images
DATASET_DIR = Path(__file__).parent / "dataset"
OUTPUT_FILE = Path(__file__).parent / "image_metadata.csv"

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def extract_features(image_path):
    """Extract a normalized RGB colour histogram."""
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((128, 128))
        pixels = np.asarray(image)

    features = []
    for channel in range(3):
        histogram, _ = np.histogram(
            pixels[:, :, channel],
            bins=16,
            range=(0, 256),
        )
        features.extend(histogram.astype(float))

    vector = np.asarray(features, dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def main():
    image_paths = sorted(
        path for path in DATASET_DIR.rglob("*")
        if path.suffix.lower() in SUPPORTED
    )

    if not image_paths:
        print("No images found in the dataset folder.")
        return

    records = []
    vectors = []

    for index, path in enumerate(image_paths, start=1):
        vector = extract_features(path)
        vectors.append(vector)
        records.append({
            "image_id": f"IMG{index:03d}",
            "filename": path.name,
            "label": path.stem.replace("_", " "),
            "source": "Provided demo dataset; original source not verified",
            "licence": "Verify source and usage rights before submission",
        })

    matrix = np.vstack(vectors)
    similarity = cosine_similarity(matrix)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Images processed: {len(records)}")
    print(f"Feature vector size per image: {matrix.shape[1]}")
    print(f"Metadata saved to: {OUTPUT_FILE.name}")
    print("\nTop matches for the first image:")

    ranked = np.argsort(similarity[0])[::-1]
    for index in ranked[:5]:
        print(
            f"{records[index]['filename']}: "
            f"similarity score = {similarity[0, index]:.3f}"
        )

    print("\nNote: similarity scores are not probabilities of correctness.")


if __name__ == "__main__":
    main()
