import os
import pickle

# Local file read is a placeholder for a future GCS fetch (e.g. downloading
# metadata.pkl from a bucket at cold start) once ingestion moves off-machine.
METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "data", "output", "metadata.pkl"
)

with open(METADATA_PATH, "rb") as f:
    _metadata = pickle.load(f)

text_metadata_df = _metadata["text_metadata_df"]
image_metadata_df = _metadata["image_metadata_df"]
