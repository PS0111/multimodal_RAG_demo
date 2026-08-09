import json
from pathlib import Path

import pytest

from multimodal_rag_demo.eval.evaluators import (
    answer_relevance,
    completeness,
    groundedness,
    retrieval_relevance,
)

GOLDEN_QUESTIONS = json.loads((Path(__file__).parent / "golden_questions.json").read_text())

# Groundedness is the only evaluator run for out_of_context questions: it's
# the only one that sees the retrieved context alongside the answer, so it's
# the only one that can verify a refusal was actually warranted rather than
# just judging (question, answer) in isolation, which can't tell a correct
# refusal from a plausible-sounding hallucination. The other three assume
# the question is answerable and are skipped for out_of_context.
EVALUATORS = {
    "retrieval_relevance": lambda case, answer, text_matches, image_matches:
        retrieval_relevance.evaluate(case["question"], text_matches, image_matches),
    "completeness": lambda case, answer, text_matches, image_matches:
        completeness.evaluate(case["question"], answer),
    "answer_relevance": lambda case, answer, text_matches, image_matches:
        answer_relevance.evaluate(case["question"], answer),
    "groundedness": lambda case, answer, text_matches, image_matches:
        groundedness.evaluate(text_matches, image_matches, answer),
}


def _build_params():
    params = []
    for case in GOLDEN_QUESTIONS:
        names = ["groundedness"] if case["category"] == "out_of_context" else list(EVALUATORS)
        params.extend(pytest.param(case, name, id=f"{case['id']}-{name}") for name in names)
    return params


@pytest.mark.llm_eval
@pytest.mark.parametrize("case, evaluator_name", _build_params())
def test_golden_question_evaluator(case, evaluator_name, qa_results):
    answer, text_matches, image_matches = qa_results(case)
    verdict = EVALUATORS[evaluator_name](case, answer, text_matches, image_matches)
    assert verdict["verdict"] == "PASS", verdict.get("reason", "")
