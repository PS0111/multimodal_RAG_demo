from multimodal_rag_demo.eval.format_context import format_image_descriptions, format_text_chunks
from multimodal_rag_demo.eval.judge_client import run_judge
from multimodal_rag_demo.eval.prompts import RETRIEVAL_RELEVANCE_PROMPT


def evaluate(question: str, text_matches: dict, image_matches: dict) -> dict:
    prompt = RETRIEVAL_RELEVANCE_PROMPT.format(
        question=question,
        text_chunks=format_text_chunks(text_matches),
        image_descriptions=format_image_descriptions(image_matches),
    )
    return run_judge(prompt)
