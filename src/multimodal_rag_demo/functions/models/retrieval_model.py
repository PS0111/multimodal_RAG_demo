from multimodal_rag_demo.data_prep.vendor import multimodal_qa_with_rag_utils as rag_utils

from multimodal_rag_demo.functions.models.metadata_store import image_metadata_df, text_metadata_df


def search_text(query: str, top_n: int = 5):
    return rag_utils.get_similar_text_from_query(
        query,
        text_metadata_df,
        column_name="text_embedding_chunk",
        top_n=top_n,
    )


def search_images(query: str, top_n: int = 5):
    return rag_utils.get_similar_image_from_query(
        text_metadata_df,
        image_metadata_df,
        query=query,
        column_name="text_embedding_from_image_description",
        image_emb=False,
        top_n=top_n,
    )
