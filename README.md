# multimodal_RAG_demo

A two-part demo: an offline data-preparation pipeline that extracts text and
image metadata (plus embeddings) from a PDF, and a FastAPI backend that
exposes retrieval over that metadata as HTTP endpoints.

## Project layout

This is a proper installable Python package (`src/` layout), not a loose
collection of scripts. Everything under `src/multimodal_rag_demo/` is one
top-level package; `tests/` and `scripts/` sit alongside it. Generated
artifacts (extracted images, `metadata.pkl`) live in a separate top-level
`data/` directory, kept out of the source tree.

```
multimodal_RAG_demo/
├── pyproject.toml
├── src/multimodal_rag_demo/
│   ├── main.py                 FastAPI app
│   ├── common/logging_config.py
│   ├── data_prep/               ingestion pipeline
│   ├── functions/                retrieval + QA backend
│   └── eval/                     LLM-judge evaluators (used by tests/)
├── tests/                        pytest suite
├── scripts/                      manual dev tools, not part of the test suite
├── data/output/                  generated: metadata.pkl, extracted images
└── logs/                         generated: one timestamped log file per run
```

### One-time setup

```bash
pip install -r requirements.txt
pip install -e .
```

The editable install (`pip install -e .`) is what makes `multimodal_rag_demo`
importable from anywhere - any working directory, any invocation style - so
none of the commands below depend on being run from a particular directory.

## Local workflow

### 1. Data preparation

Set `PDF_PATH` in
[`src/multimodal_rag_demo/data_prep/config.py`](src/multimodal_rag_demo/data_prep/config.py)
to point at the PDF you want to ingest, then run:

```bash
python -m multimodal_rag_demo.data_prep.run_ingestion
```

This extracts per-page text and embedded images from the PDF, generates
embeddings and Gemini-authored image descriptions, and writes the resulting
`text_metadata_df` / `image_metadata_df` DataFrames to `data/output/metadata.pkl`.
Extracted images are saved under `data/output/images/`.

### 2. Retrieval backend

```bash
uvicorn multimodal_rag_demo.main:app --reload
```

Then hit the two debug endpoints to inspect retrieval quality against the
metadata produced in step 1:

```
GET /debug/search-text?query=<text>&top_n=<int>
GET /debug/search-image?query=<text>&top_n=<int>
```

Each returns JSON citations and also logs the formatted citation
server-side.

For a full answer synthesized from both retrieved text and image context, use:

```
GET /query?query=<text>
```

This returns the generated answer as a plain string. The QA instruction
prompt lives in
[`src/multimodal_rag_demo/functions/prompts.py`](src/multimodal_rag_demo/functions/prompts.py),
separate from the image-description prompt used during ingestion.

### 3. Evaluation

`tests/` is a real pytest suite scoring the QA pipeline against a fixed set
of 20 golden questions
([`tests/golden_questions.json`](tests/golden_questions.json)) covering
text-factual, image-factual, reasoning, and out-of-context cases. The
evaluator implementations themselves live in
[`src/multimodal_rag_demo/eval/`](src/multimodal_rag_demo/eval/) (source
code, not test code); `tests/test_eval_golden_questions.py` just exercises
them.

```bash
pytest -m llm_eval
```

This is tagged with a custom marker (`llm_eval`) because it calls real
Gemini APIs and costs real time/money - it will *not* run under a bare
`pytest` (which by default runs everything pytest discovers, including
this), so use `pytest -m llm_eval` explicitly. `tests/conftest.py` handles
one-time setup (`init_vertex()`) via a session-scoped fixture, and caches
each question's generated answer so it's computed once and reused across
that question's evaluator test cases, not re-generated per evaluator.

For each golden question, one independent pytest test node runs per
applicable evaluator (`pytest -v` shows them individually, e.g.
`test_golden_question_evaluator[text_factual_01-groundedness]`), each a
binary PASS/FAIL:

- **Retrieval relevance** - is the retrieved context relevant to the
  question?
- **Groundedness** - is the answer supported by the retrieved context, with
  no hallucination?
- **Completeness** - does the answer address every part of the question?
- **Answer relevance** - is the answer on-topic for the question asked?

`out_of_context` questions only run the groundedness test. It's the only
evaluator that sees the retrieved context alongside the answer, so it's the
only one that can verify a refusal was actually warranted rather than just
judging `(question, answer)` in isolation - which can't tell a correct
refusal apart from a plausible-sounding hallucination. Groundedness
explicitly treats a correct refusal ("not enough context to answer") as a
PASS for any category; retrieval relevance, completeness, and answer
relevance all assume the question is answerable, so a refusal should FAIL
them - which is also why they're skipped for `out_of_context`, where a
refusal is the *correct* behavior. The judge itself uses a different
(newer) Gemini model than the one used for answer generation - see
`JUDGE_MODEL_NAME` in
[`src/multimodal_rag_demo/eval/judge_client.py`](src/multimodal_rag_demo/eval/judge_client.py)
- to avoid the model grading its own answers.

### Manual dev tools

[`scripts/caption_single_image.py`](scripts/caption_single_image.py)
re-runs image captioning on one image in isolation (without a full
ingestion pass), useful for inspecting a suspect caption's logged
`finish_reason`. Not part of the pytest suite - run directly:

```bash
python scripts/caption_single_image.py
```

### Logging

[`src/multimodal_rag_demo/common/logging_config.py`](src/multimodal_rag_demo/common/logging_config.py)
configures a shared `logger` as an import side effect - `from
multimodal_rag_demo.common.logging_config import logger` is the entire
integration surface. Each process run writes to its own timestamped file
under `logs/`, plus the console.

## Not included in this pass

- Page-level PNG rendering (only embedded raster images are extracted)
- S3/GCS-driven ingestion (PDF path is a local, hardcoded config value)
- A unified single-call text+image search endpoint
- A vector database (retrieval scans the in-memory DataFrames directly)
