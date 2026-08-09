from multimodal_rag_demo.common import vertex_client
from multimodal_rag_demo.data_prep.config import EMBEDDING_SIZE, IMAGE_SAVE_DIR, PDF_PATH
from multimodal_rag_demo.data_prep.prompts import IMAGE_DESCRIPTION_PROMPT
from multimodal_rag_demo.data_prep.vendor import multimodal_qa_with_rag_utils as rag_utils


def extract_metadata():
    vertex_client.init_vertex()

    text_metadata_df, image_metadata_df = rag_utils.get_document_metadata(
        vertex_client.generative_multimodal_model,
        pdf_path=PDF_PATH,
        image_save_dir=IMAGE_SAVE_DIR,
        image_description_prompt=IMAGE_DESCRIPTION_PROMPT,
        embedding_size=EMBEDDING_SIZE,
    )

    return text_metadata_df, image_metadata_df
