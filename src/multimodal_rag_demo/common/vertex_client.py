import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel

from multimodal_rag_demo.data_prep.config import LOCATION, PROJECT_ID
from multimodal_rag_demo.data_prep.vendor import multimodal_qa_with_rag_utils as rag_utils

# Single source of truth for Vertex AI client setup, shared by data_prep
# (ingestion/captioning) and functions (live serving). Populated by
# init_vertex(); modules that need the Gemini model must read this attribute
# at call time (not import it by name), since it is still None until
# init_vertex() has run.
generative_multimodal_model: GenerativeModel | None = None


def init_vertex() -> None:
    """Initializes the Vertex AI SDK and the models this project depends on.

    Must run exactly once, before any generation/retrieval/ingestion call.
    Same generative model (gemini-2.5-flash) is used for both image
    captioning at ingestion time and live QA answer generation, so the two
    paths are always tested against the same model. The eval harness's
    judge model (gemini-2.5-pro, see eval/judge_client.py) is deliberately
    kept separate from this, so the judge isn't grading its own homework.
    """
    global generative_multimodal_model

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    rag_utils.set_global_variable(
        "text_embedding_model", TextEmbeddingModel.from_pretrained("text-embedding-004")
    )
    rag_utils.set_global_variable(
        "multimodal_embedding_model",
        MultiModalEmbeddingModel.from_pretrained("multimodalembedding"),
    )

    generative_multimodal_model = GenerativeModel("gemini-2.5-flash")
