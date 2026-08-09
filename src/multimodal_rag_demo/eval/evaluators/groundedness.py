from multimodal_rag_demo.eval.format_context import format_image_descriptions, format_text_chunks
from multimodal_rag_demo.eval.judge_client import run_judge
from multimodal_rag_demo.eval.prompts import GROUNDEDNESS_PROMPT


def evaluate(text_matches: dict, image_matches: dict, answer: str) -> dict:
    prompt = GROUNDEDNESS_PROMPT.format(
        text_chunks=format_text_chunks(text_matches),
        image_descriptions=format_image_descriptions(image_matches),
        answer=answer,
    )
    return run_judge(prompt)
