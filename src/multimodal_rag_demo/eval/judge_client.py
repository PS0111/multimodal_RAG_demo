import json

from vertexai.generative_models import GenerationConfig, GenerativeModel

# A step up from the gemini-2.5-flash used for answer generation (see
# multimodal_rag_demo/functions/vertex_client.py), so the judge isn't
# grading its own homework with the exact same (lighter) model. Assumes
# vertexai.init() has already run (tests/conftest.py's session fixture
# calls multimodal_rag_demo.functions.vertex_client.init_vertex() at startup).
JUDGE_MODEL_NAME = "gemini-2.5-pro"

# Constrains generate_content() to this exact shape instead of relying on the
# model following free-text formatting instructions in the prompt.
VERDICT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": ["PASS", "FAIL"]},
        "reason": {"type": "STRING"},
    },
    "required": ["verdict", "reason"],
}

# No max_output_tokens cap: this SDK has no separate thinking-token budget
# (no ThinkingConfig, no such field on GenerationConfig) - max_output_tokens
# is the only lever that exists, and it governs one shared pool for both
# internal thinking and the final visible answer. Capping it caps thinking,
# which is not what we want, so it's left uncapped.

_judge_model: GenerativeModel | None = None


def _get_judge_model() -> GenerativeModel:
    global _judge_model
    if _judge_model is None:
        _judge_model = GenerativeModel(JUDGE_MODEL_NAME)
    return _judge_model


def run_judge(prompt: str) -> dict:
    """Calls the judge model and parses its structured PASS/FAIL verdict.

    Never raises - an API failure or an unparseable response is logged and
    turned into a {"verdict": "ERROR", ...} result instead, so one failed
    judge call doesn't halt the rest of the eval run.
    """
    try:
        response = _get_judge_model().generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=VERDICT_SCHEMA,
            ),
        )
    except Exception as e:
        print(f"  [judge_client] generate_content() failed: {e!r}", flush=True)
        return {"verdict": "ERROR", "reason": f"judge call failed: {e}"}

    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, ValueError) as e:
        print(
            f"  [judge_client] failed to parse judge response: {e!r} - raw response.text={response.text!r}",
            flush=True,
        )
        return {"verdict": "ERROR", "reason": f"failed to parse judge response: {e}"}
