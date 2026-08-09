import os
import pickle

from multimodal_rag_demo.data_prep.extract_metadata import extract_metadata

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "metadata.pkl")


def main():
    text_metadata_df, image_metadata_df = extract_metadata()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(
            {"text_metadata_df": text_metadata_df, "image_metadata_df": image_metadata_df},
            f,
        )

    print(f"Wrote metadata for {len(text_metadata_df)} text chunks and "
          f"{len(image_metadata_df)} images to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
