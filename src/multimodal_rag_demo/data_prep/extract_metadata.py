import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel

from multimodal_rag_demo.data_prep.config import (
    EMBEDDING_SIZE,
    IMAGE_SAVE_DIR,
    LOCATION,
    PDF_PATH,
    PROJECT_ID,
)
from multimodal_rag_demo.data_prep.prompts import IMAGE_DESCRIPTION_PROMPT
from multimodal_rag_demo.data_prep.vendor import multimodal_qa_with_rag_utils as rag_utils


def extract_metadata():
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    generative_multimodal_model = GenerativeModel("gemini-2.5-pro")

    # The vendored functions look up these embedding models as globals within
    # their own module, so they must be registered there via set_global_variable.
    rag_utils.set_global_variable(
        "text_embedding_model", TextEmbeddingModel.from_pretrained("text-embedding-004")
    )
    rag_utils.set_global_variable(
        "multimodal_embedding_model",
        MultiModalEmbeddingModel.from_pretrained("multimodalembedding"),
    )

    text_metadata_df, image_metadata_df = rag_utils.get_document_metadata(
        generative_multimodal_model,
        pdf_path=PDF_PATH,
        image_save_dir=IMAGE_SAVE_DIR,
        image_description_prompt=IMAGE_DESCRIPTION_PROMPT,
        embedding_size=EMBEDDING_SIZE,
    )

    return text_metadata_df, image_metadata_df
