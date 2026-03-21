# CS4120-First-Aid-Co-Pilot

First-Aid Co-Pilot is a sparse-RAG first-aid assistant built on top of the cleaned dataset artifacts in `preprocessing/`. It uses:

- a custom TF-IDF + cosine similarity retriever
- a file-backed sparse vector store
- a single LangChain retrieval tool, `search_first_aid_knowledge`
- `langchain.agents.create_agent`
- Ollama-backed local models
- FastAPI for the HTTP layer
- a small CLI for development and local use

The project is designed to go from the existing preprocessing outputs through a usable retrieval + agentic pipeline. Full evaluation and visualization are intentionally out of scope in the current implementation.

## What The Project Does

At runtime, the system:

1. loads a persisted TF-IDF index built from the first-aid QA corpus
2. retrieves the most relevant question-answer documents with cosine similarity
3. exposes retrieval through a LangChain tool, `search_first_aid_knowledge`
4. runs a local Ollama model through `create_agent`
5. applies deterministic safety checks for high-risk emergency categories
6. logs conversation turns and run metadata as JSONL files

If Ollama is unavailable, the project does not crash. It falls back to a retrieval-backed answer and returns warnings explaining that the agent call failed.

## Repository Layout

```text
preprocessing/                 Input CSVs and notebook from the preprocessing phase
firstaid_copilot/             Application package
tests/                        Unit and integration-style tests
artifacts/indexes/            Persisted TF-IDF indexes
artifacts/conversations/      JSONL conversation logs and run telemetry
```

Important source files:

- [firstaid_copilot/data.py](F:/Private/Personal/me/CS4120-First-Aid-Co-Pilot/firstaid_copilot/data.py)
- [firstaid_copilot/tuning.py](F:/Private/Personal/me/CS4120-First-Aid-Co-Pilot/firstaid_copilot/tuning.py)
- [firstaid_copilot/vector_store.py](F:/Private/Personal/me/CS4120-First-Aid-Co-Pilot/firstaid_copilot/vector_store.py)
- [firstaid_copilot/service.py](F:/Private/Personal/me/CS4120-First-Aid-Co-Pilot/firstaid_copilot/service.py)
- [firstaid_copilot/api.py](F:/Private/Personal/me/CS4120-First-Aid-Co-Pilot/firstaid_copilot/api.py)
- [firstaid_copilot/cli.py](F:/Private/Personal/me/CS4120-First-Aid-Co-Pilot/firstaid_copilot/cli.py)

## Data Inputs

The implementation assumes preprocessing is already done and uses these files from `preprocessing/`:

- `train.csv`
- `dev.csv`
- `test.csv`
- `full_clean.csv`
- `eval_subset.csv`
- `robustness_test.csv`

Retrieval indexes are built only from:

- `train.csv` for the `experiment` profile
- `full_clean.csv` for the `demo` profile

The evaluation subset and robustness file are preserved for future work but are not included in the retrieval corpus.

## Retrieval Design

Each QA row is converted into one LangChain `Document`.

The indexed text template is:

```text
Question: {question}
Question: {question}
Answer: {answer}
Category: {category}
Source: {source}
```

Each document also stores metadata:

- `doc_id`
- `question`
- `answer`
- `question_norm`
- `category`
- `source`
- `split`

Retrieval uses:

- TF-IDF vectorization
- cosine similarity
- persisted sparse matrices on disk
- no dense embeddings
- no vector database
- no hybrid search

### TF-IDF Tuning

When an index is built, the project tunes TF-IDF parameters against the `train/dev` setup and persists the winning configuration to `artifacts/indexes/<profile>/config.json`.

The tuning grid is:

- `ngram_range`: `(1,1)`, `(1,2)`
- `min_df`: `1`, `2`, `3`
- `max_df`: `0.95`, `1.0`
- `sublinear_tf`: `False`, `True`
- `max_features`: `None`, `10000`, `20000`
- `stop_words`: `None`
- `norm`: `l2`

The selection objective combines:

- top-1 answer unigram F1
- best top-3 answer unigram F1
- category hit@1
- category hit@3

The selected parameters are not recomputed at query time. They are persisted and reused.

## Profiles

The project defines two index profiles.

### `experiment`

- built from `preprocessing/train.csv`
- intended for tuning and development checks
- keeps retrieval work aligned with the train/dev split

### `demo`

- built from `preprocessing/full_clean.csv`
- intended for the actual interactive assistant
- uses the full cleaned corpus for broader retrieval coverage

If you just want to use the assistant, use `demo`.

## Supported Models

The current configured model set is:

- `functiongemma`
- `qwen3:0.6b`
- `granite4:350m`

Notes:

- Ollama often reports FunctionGemma as `functiongemma:latest`
- the project resolves `functiongemma` against `functiongemma:latest` automatically
- if `granite4:350m` has not been pulled locally, it will show as unavailable in `doctor` and `/models`

## Prerequisites

### Python

Use Python 3.13 or similar modern Python on Windows.

### Ollama

Ollama must be installed and running if you want the full agent path.

The application checks Ollama at:

- `http://localhost:11434`

You can override this with:

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434"
```

The default model for query requests can be overridden with:

```powershell
$env:FIRSTAID_DEFAULT_MODEL="qwen3:0.6b"
```

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Ollama Setup

If you want the full local agent flow, start Ollama and pull the models you plan to use.

Examples:

```powershell
ollama pull functiongemma
ollama pull qwen3:0.6b
ollama pull granite4:350m
```

If Ollama serves FunctionGemma as `functiongemma:latest`, that is fine. The project handles that alias.

## Build Indexes

Build the experiment index:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot build-index --profile experiment
```

Build the demo index:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot build-index --profile demo
```

What happens during index build:

- the project loads the preprocessing CSVs
- tunes TF-IDF hyperparameters against `train/dev`
- builds the requested document corpus
- serializes the vectorizer, sparse matrix, documents, and config to disk

## CLI Usage

### `doctor`

Checks:

- Python executable
- whether `.venv` exists
- whether Ollama is reachable
- which configured models are available
- whether the `experiment` and `demo` indexes are built

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot doctor
```

### `build-index`

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot build-index --profile experiment
.\.venv\Scripts\python.exe -m firstaid_copilot build-index --profile demo
```

Force a rebuild:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot build-index --profile demo --force
```

### `query`

Basic usage:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot query --model qwen3:0.6b --profile demo --text "Someone is choking and cannot speak"
```

Optional flags:

- `--model`
- `--profile`
- `--text`
- `--top-k`
- `--session-id`

Example with explicit retrieval depth:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot query --model functiongemma --profile demo --top-k 3 --text "How should I help someone with severe bleeding?"
```

### `serve`

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot serve --host 127.0.0.1 --port 8000
```

## API

Once the server is running:

```text
GET  /health
GET  /models
POST /retrieve
POST /query
```

### `GET /health`

Returns:

- application status
- Ollama availability
- built profiles
- configured model availability

### `GET /models`

Returns the configured model list and whether each model is available locally through Ollama.

### `POST /retrieve`

Request body:

```json
{
  "query": "Someone is choking and cannot speak",
  "profile": "demo",
  "k": 5
}
```

Response body shape:

```json
{
  "query": "Someone is choking and cannot speak",
  "profile": "demo",
  "hits": [
    {
      "doc_id": "full_clean-01769",
      "question": "...",
      "answer": "...",
      "category": "choking",
      "source": "FirstAidQA",
      "split": "full_clean",
      "score": 0.377179
    }
  ]
}
```

### `POST /query`

Request body:

```json
{
  "query": "Someone is choking and cannot speak",
  "model": "qwen3:0.6b",
  "profile": "demo",
  "top_k": 3,
  "session_id": "optional-session-id"
}
```

Response fields:

- `session_id`
- `turn_id`
- `query`
- `model`
- `profile`
- `risk_category`
- `call_emergency_now`
- `steps`
- `answer_text`
- `sources`
- `retrieval_hits`
- `warnings`
- `used_retrieval_tool`

PowerShell example:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/query `
  -ContentType "application/json" `
  -Body '{"query":"Someone is choking and cannot speak","model":"qwen3:0.6b","profile":"demo","top_k":3}'
```

## Runtime Behavior

### Retrieval-first agent behavior

The system prompt explicitly instructs the agent to:

- call `search_first_aid_knowledge` before answering
- stay grounded in retrieved content
- answer in concise numbered steps
- mention emergency escalation early for high-risk scenarios

### Safety behavior

The project applies deterministic safety handling for known high-risk categories such as:

- choking
- CPR
- severe bleeding
- stroke
- heart attack
- unconsciousness
- spinal injury

If emergency escalation language is missing from a high-risk answer, the service retries once with stricter instructions. If needed, it injects emergency guidance outside the model response.

### Fallback behavior

If Ollama is down or the agent call fails:

- retrieval still runs
- the system returns a retrieval-backed fallback answer
- the response includes warnings describing the failure

This means the project can still produce grounded retrieval results even when model serving is unavailable.

## Storage

### Index storage

Indexes are written under:

- `artifacts/indexes/experiment/`
- `artifacts/indexes/demo/`

Each profile stores:

- `vectorizer.joblib`
- `doc_matrix.npz`
- `documents.jsonl`
- `config.json`

### Conversation storage

Conversation logs are written under:

- `artifacts/conversations/`

Files include:

- `session-<timestamp>-<id>.jsonl`
- `runs.jsonl`

Each logged turn includes:

- `session_id`
- `turn_id`
- `timestamp`
- `user_query`
- `model`
- `profile`
- `risk_category`
- `retrieval_hits`
- `final_answer`
- `warnings`

## Tests

Run the test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The test suite covers:

- data loading and document creation
- TF-IDF tuning and vector store persistence
- safety checks
- API shapes
- service-level retrieval and model alias handling

## Troubleshooting

### `doctor` says Ollama is unavailable

Check that Ollama is running and reachable at `OLLAMA_BASE_URL`.

### `functiongemma` looks unavailable even though it is installed

Ollama may expose it as `functiongemma:latest`. The project now maps that automatically. Re-run:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot doctor
```

### `granite4:350m` shows unavailable

It has not been pulled yet. Pull it with:

```powershell
ollama pull granite4:350m
```

### Query returns a fallback answer with warnings

That usually means:

- Ollama is not running
- the requested model is not available locally
- the model invocation failed and the service fell back to retrieval-only output

### Index is missing

Run:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot build-index --profile experiment
.\.venv\Scripts\python.exe -m firstaid_copilot build-index --profile demo
```
