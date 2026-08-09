from fastapi import APIRouter

from multimodal_rag_demo.functions.models import qa_model

router = APIRouter()


@router.get("/query")
def ask(query: str) -> str:
    return qa_model.answer_query(query)
