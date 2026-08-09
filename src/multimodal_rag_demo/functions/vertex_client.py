import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

from multimodal_rag_demo.data_prep.config import LOCATION, PROJECT_ID
from multimodal_rag_demo.data_prep.vendor import multimodal_qa_with_rag_utils as rag_utils

# Populated by init_vertex() at application startup. Modules that need the
# Gemini model must read this attribute at call time (not import it by name),
# since it is still None until the app's lifespan startup hook has run.
generative_multimodal_model: GenerativeModel | None = None


def init_vertex() -> None:
    """Initializes the Vertex AI SDK and the models this service depends on.

    Must run exactly once, at application startup, before any request is served.
    """
    global generative_multimodal_model

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    rag_utils.set_global_variable(
        "text_embedding_model", TextEmbeddingModel.from_pretrained("text-embedding-004")
    )

    # Lighter/faster model than the gemini-2.5-pro used in
    # data_prep/extract_metadata.py for image description generation at
    # ingestion time - answer generation is on the live request path.
    generative_multimodal_model = GenerativeModel("gemini-2.5-flash")
