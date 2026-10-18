# Image Similarity Search for a Small Dataset

## What this project does
Upload a query image and the app ranks images in the included demo dataset using RGB colour histograms and cosine similarity.

## Run on your laptop
1. Install Python 3.10 or newer from https://www.python.org/downloads/
2. Extract this ZIP folder.
3. Open Command Prompt/Terminal in the extracted `image_similarity_capstone` folder.
4. Create a virtual environment (recommended):
   - Windows: `py -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Run: `streamlit run app.py`
7. Open the local URL printed in the terminal (usually http://localhost:8501).

## Included
- `app.py`: Streamlit web app and feature extraction
- `dataset/`: small synthetic demo image collection
- `requirements.txt`: Python dependencies

## Important for your submission
- This is a working baseline prototype, not a production image search engine.
- Replace/add properly licensed images if your assignment requires an externally sourced dataset, and record each source and licence.
- Take screenshots of the home page, uploaded query, ranked results and score display.
- Prepare a short evaluation table with query image, relevant top-3 results, observations and limitations.
- Do not claim the scores are probabilities or claim high accuracy without a measured evaluation.
