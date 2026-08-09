from multimodal_rag_demo.data_prep.vendor import multimodal_qa_with_rag_utils as rag_utils

from multimodal_rag_demo.common import vertex_client
from multimodal_rag_demo.functions.models.metadata_store import image_metadata_df, text_metadata_df
from multimodal_rag_demo.functions.prompts import QA_PROMPT


def answer_query_with_context(query: str) -> tuple:
    """Returns (answer, retrieved_text_chunks, retrieved_image_descriptions).

    Used by the eval harness (tests/test_eval_golden_questions.py), which
    needs the exact context that was fed to the generator to score
    groundedness/retrieval relevance. The production endpoint only needs
    the answer string.
    """
    return rag_utils.get_answer_from_qa_system(
        query,
        text_metadata_df,
        image_metadata_df,
        instruction=QA_PROMPT,
        model=vertex_client.generative_multimodal_model,
    )


def answer_query(query: str) -> str:
    response, _, _ = answer_query_with_context(query)
    return response
