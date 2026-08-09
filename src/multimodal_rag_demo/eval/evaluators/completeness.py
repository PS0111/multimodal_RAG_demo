from multimodal_rag_demo.eval.judge_client import run_judge
from multimodal_rag_demo.eval.prompts import COMPLETENESS_PROMPT


def evaluate(question: str, answer: str) -> dict:
    prompt = COMPLETENESS_PROMPT.format(question=question, answer=answer)
    return run_judge(prompt)
