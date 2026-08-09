import pytest

from multimodal_rag_demo.functions import vertex_client
from multimodal_rag_demo.functions.models import qa_model


@pytest.fixture(scope="session", autouse=True)
def _init_vertex():
    vertex_client.init_vertex()


@pytest.fixture(scope="session")
def qa_results():
    """Caches (answer, text_matches, image_matches) per golden question id
    for the whole test session, so the expensive QA generation call runs
    once per question, not once per (question, evaluator) test instance."""
    cache = {}

    def _get(case):
        if case["id"] not in cache:
            cache[case["id"]] = qa_model.answer_query_with_context(case["question"])
        return cache[case["id"]]

    return _get
