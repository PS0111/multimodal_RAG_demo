"""
Re-runs image captioning on one suspect image in isolation, without
re-running the full ingestion pass over the whole deck, so its logged
finish_reason (from get_gemini_response) can be inspected directly.
"""

import vertexai
from vertexai.generative_models import GenerativeModel, Image

from multimodal_rag_demo.common.logging_config import logger
from multimodal_rag_demo.data_prep.config import LOCATION, PROJECT_ID
from multimodal_rag_demo.data_prep.prompts import IMAGE_DESCRIPTION_PROMPT
from multimodal_rag_demo.data_prep.vendor import multimodal_qa_with_rag_utils as rag_utils

# Point this at a specific extracted image saved under IMAGE_SAVE_DIR from a
# prior ingestion run, e.g. the Chaebol diagram from page 44.
IMAGE_PATH = "data/output/images/MBA_2025_session_1_intro_CS.pdf_image_43_1_248.jpeg"


def main():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    generative_multimodal_model = GenerativeModel("gemini-2.5-pro")

    image_for_gemini = Image.load_from_file(IMAGE_PATH)

    caption = rag_utils.get_gemini_response(
        generative_multimodal_model,
        model_input=[IMAGE_DESCRIPTION_PROMPT, image_for_gemini],
        stream=True,
    )

    logger.info(f"Caption text ({len(caption)} chars): {caption}")


if __name__ == "__main__":
    main()
