from multimodal_rag_demo.eval.judge_client import run_judge
from multimodal_rag_demo.eval.prompts import ANSWER_RELEVANCE_PROMPT


def evaluate(question: str, answer: str) -> dict:
    prompt = ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
    return run_judge(prompt)
