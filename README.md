# CS4120 First-Aid Co-Pilot

First-Aid Co-Pilot is a local sparse-RAG assistant for first-aid question answering. It retrieves relevant guidance from a cleaned FirstAidQA corpus with TF-IDF + cosine similarity, then uses a small local Ollama model to produce a concise answer with safety checks.

The system is designed to run without a hosted LLM API. If Ollama is unavailable or a model call fails, retrieval can still run and the app falls back to retrieved first-aid guidance instead of returning an unsupported model-only answer.

## Quick Start

These commands assume Windows PowerShell and that you are running them from the repository root. The project was developed and tested with Python 3.12 on Windows.

```powershell
git clone https://github.com/AnanyaS05/CS4120-First-Aid-Co-Pilot.git
cd CS4120-First-Aid-Co-Pilot
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you downloaded the project as a ZIP, unzip it and `cd` into the extracted folder instead of running `git clone`.

### 1. Confirm Data Files

The primary QA corpus comes from the FirstAidQA dataset on Hugging Face:
`https://huggingface.co/datasets/i-am-mushfiq/FirstAidQA`.

The preprocessed CSV files needed to run the app are committed in this repository. A fresh clone should already contain these files under `preprocessing/`:

```text
preprocessing/train.csv
preprocessing/dev.csv
preprocessing/test.csv
preprocessing/full_clean.csv
preprocessing/generated_answer_eval.csv
```

The main QA splits should include:

```text
question, answer, source, question_norm, category
```

To regenerate the preprocessed files from the source dataset, run the notebook:

```text
preprocessing/firstaid_preprocessing.ipynb
```

The notebook documents the original preprocessing workflow and writes the CSV splits used by the application. For normal reproduction, you do not need to rerun the notebook unless you intentionally want to rebuild the dataset splits.

### Generated Artifacts

The `artifacts/` directory is generated at runtime and is intentionally ignored by git. A fresh clone will not contain retrieval indexes, evaluation outputs, or conversation logs. These are recreated by the commands below:

```text
artifacts/indexes/          created by build-index
artifacts/evaluations/      created by evaluate / evaluate-tfidf
artifacts/conversations/    created by CLI, API, or UI queries
```

If you want a completely fresh run, delete generated artifacts first:

```powershell
Remove-Item -Recurse -Force .\artifacts\indexes -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\artifacts\evaluations -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\artifacts\conversations -ErrorAction SilentlyContinue
```

### 2. Install And Start Ollama

Install Ollama from `https://ollama.com/`, start it, then pull at least one supported model:

```powershell
ollama pull qwen3.5:0.8b
```

Optional, pull all configured models:

```powershell
ollama pull functiongemma
ollama pull qwen3:0.6b
ollama pull qwen3.5:0.8b
ollama pull granite4:350m
```

The app uses Ollama at `http://localhost:11434` by default. To override it:

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434"
```

The UI, CLI, and API let you choose the model explicitly. The examples below use `qwen3.5:0.8b`.

### 3. Build The Indexes

Build both retrieval profiles before running the UI:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot build-index --profile experiment --force
.\.venv\Scripts\python.exe -m firstaid_copilot build-index --profile demo --force
```

What this does:

- loads the preprocessed CSV files
- tunes TF-IDF parameters with the train/dev/test setup
- builds the requested retrieval corpus
- saves the vectorizer, sparse matrix, documents, and config under `artifacts/indexes/`

Profiles:

- `experiment`: indexes `train.csv`; useful for development and evaluation
- `demo`: indexes `full_clean.csv`; used by the web UI for broader coverage

### 4. Check The Environment

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot doctor
```

This reports:

- whether Ollama is reachable
- which configured models are locally available
- whether the `experiment` and `demo` indexes exist
- whether the project virtual environment is present

### 5. Run The Web UI

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot serve --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Use the `demo` profile for normal interaction. Previous chats are saved under `artifacts/conversations/`.

## Running From The CLI

Non-streaming query:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot query --model qwen3.5:0.8b --profile demo --top-k 5 --text "Someone is choking and cannot speak"
```

Streaming query:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot query --model qwen3.5:0.8b --profile demo --top-k 5 --stream --text "Someone is choking and cannot speak"
```

If Ollama is down or the selected model fails, the service attempts a retrieval-backed fallback and includes warnings in the response.

## Running Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover:

- data loading and document creation
- TF-IDF tuning and vector store persistence
- safety checks
- API response shapes
- service-level retrieval, model alias handling, retry/fallback, and streaming behavior
- CLI streaming rendering

## Running Evaluation

Run only TF-IDF selection:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot evaluate-tfidf
```

Run the final evaluation:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot evaluate --profile demo --top-k 5
```

Final evaluation writes artifacts under:

```text
artifacts/evaluations/final/
```

Generated-answer evaluation can be slow because it calls local Ollama models over `generated_answer_eval.csv`. To do a quick smoke test:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot evaluate --profile demo --top-k 5 --limit 3
```

To skip generated answers and evaluate only retrieval:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot evaluate --skip-generated
```

To reproduce the complete four-model generated-answer evaluation, first pull all configured models, then run:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot evaluate `
  --models functiongemma qwen3:0.6b qwen3.5:0.8b granite4:350m `
  --profile demo `
  --top-k 5 `
  --request-timeout-seconds 20
```

Useful evaluation options:

- `--limit N`: evaluate only the first `N` generated-answer rows per model for a quick smoke test
- `--skip-generated`: run TF-IDF retrieval evaluation only
- `--skip-unavailable-models`: skip models that Ollama does not report as available
- `--request-timeout-seconds N`: set the per-request Ollama timeout
- `--no-force-index`: reuse an existing index instead of rebuilding it before generated-answer evaluation

The evaluator checkpoints generated-answer rows to `artifacts/evaluations/final/generated_answer_rows.csv`, so rerunning the same command can resume already completed model/question rows.

## API Endpoints

Once the server is running:

```text
GET  /health
GET  /models
GET  /conversations
GET  /conversations/{session_id}
POST /retrieve
POST /query
POST /query/stream
```

Example retrieval request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/retrieve `
  -ContentType "application/json" `
  -Body '{"query":"Someone is choking and cannot speak","profile":"demo","k":5}'
```

Example query request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/query `
  -ContentType "application/json" `
  -Body '{"query":"Someone is choking and cannot speak","model":"qwen3.5:0.8b","profile":"demo","top_k":5}'
```

The streaming endpoint returns `text/event-stream` events:

```text
session, status, retrieval, token, warning, final, error
```

## Project Structure

```text
firstaid_copilot/
  api.py              FastAPI app and web routes
  cli.py              Command-line interface
  config.py           Runtime paths, profiles, and model config
  conversation.py     JSONL conversation logging
  data.py             CSV loading and document construction
  evaluation.py       Final retrieval and generated-answer evaluation
  safety.py           Deterministic emergency checks
  schemas.py          Pydantic request/response models
  service.py          Retrieval, agent, retry, fallback, and logging orchestration
  tuning.py           TF-IDF search and scoring
  vector_store.py     File-backed TF-IDF vector store
  static/             Browser UI

preprocessing/        Preprocessed CSV inputs
tests/                Unit and integration-style tests
artifacts/            Generated indexes, evaluations, and conversations
```

## How Retrieval Works

Each QA row is converted into a LangChain `Document`.

The indexed text template is:

```text
Question: {question}
Question: {question}
Answer: {answer}
Category: {category}
Source: {source}
```

The question is repeated intentionally so query wording has more weight in sparse retrieval.

Retrieval uses:

- `TfidfVectorizer`
- cosine similarity
- persisted sparse matrices
- no vector database
- no dense embeddings

## TF-IDF Tuning

When an index is built, the system tunes TF-IDF parameters and saves the selected config to:

```text
artifacts/indexes/<profile>/config.json
```

The search grid includes:

- `ngram_range`: `(1,1)`, `(1,2)`
- `min_df`: `1`, `2`, `3`
- `max_df`: `0.95`, `1.0`
- `sublinear_tf`: `False`, `True`
- `max_features`: `None`, `2000`, `2500`, `3000`
- `stop_words`: `None`
- `norm`: `l2`

The selection score combines:

- top-1 answer unigram F1
- best top-3 answer unigram F1
- category hit@1
- category hit@3

## Safety Behavior

The service uses deterministic keyword-based safety checks for high-risk categories such as:

- choking
- CPR
- severe bleeding
- stroke
- heart attack
- unconsciousness
- spinal injury

If a high-risk answer does not include emergency escalation language, the service retries once with stricter instructions. If the retry is still missing that language, it injects emergency guidance outside the model response.

This project is for educational use and is not a replacement for professional medical advice or emergency services.

## Troubleshooting

### `doctor` says Ollama is unavailable

Make sure Ollama is running and reachable:

```powershell
ollama list
```

If needed, set:

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434"
```

### A model is unavailable

Pull it with Ollama:

```powershell
ollama pull qwen3.5:0.8b
```

FunctionGemma may appear as `functiongemma:latest`; the app handles that alias automatically.

### Index files are missing

Rebuild both profiles:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot build-index --profile experiment --force
.\.venv\Scripts\python.exe -m firstaid_copilot build-index --profile demo --force
```

### `test.csv` is missing `category`

Regenerate or restore `preprocessing/test.csv` from the project data. The app needs the `category` column for TF-IDF tuning and evaluation.

### Port 8000 is already in use

Use another port:

```powershell
.\.venv\Scripts\python.exe -m firstaid_copilot serve --host 127.0.0.1 --port 8001
```

Then open:

```text
http://127.0.0.1:8001
```
