from fastapi import APIRouter

from multimodal_rag_demo.functions.models import retrieval_model
from multimodal_rag_demo.functions.models.retrieval_model import rag_utils
from multimodal_rag_demo.functions.views import citation_view

router = APIRouter()


@router.get("/debug/search-text")
def search_text(query: str, top_n: int = 5):
    results = retrieval_model.search_text(query, top_n=top_n)
    rag_utils.print_text_to_text_citation(results, print_top=False)

    return citation_view.format_text_citations(results)


@router.get("/debug/search-image")
def search_image(query: str, top_n: int = 5):
    results = retrieval_model.search_images(query, top_n=top_n)
    rag_utils.print_text_to_image_citation(results, print_top=False)

    return citation_view.format_image_citations(results)
