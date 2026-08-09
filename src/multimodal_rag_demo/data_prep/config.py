import os

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")

# TODO: PDF_PATH will become an S3/GCS-driven parameter later; hardcoded for local iteration.
PDF_PATH = "/Users/pulkitsuhasaria/Documents/mba/ESADE/program/Term_2/Part_1/Corporate_Strategy/MBA_2025_session_1_intro_CS.pdf"

IMAGE_SAVE_DIR = os.path.join(_PROJECT_ROOT, "data", "output", "images")

EMBEDDING_SIZE = 1408

PROJECT_ID = "project-9eb011be-c81e-4d01-a7c"
LOCATION = "us-central1"
