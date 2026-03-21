from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from .config import AppConfig, ProfileName, normalize_model_name
from .conversation import (
    ConversationLogger,
    make_session_id,
    make_turn_id,
    sanitize_identifier,
    utc_now_iso,
)
from .data import (
    build_documents,
    get_profile_source_split,
    load_split_dataframe,
)
from .safety import assess_query, has_required_emergency_language
from .schemas import (
    DoctorReport,
    HealthResponse,
    ModelStatus,
    QueryRequest,
    QueryResponse,
    RetrievalHit,
)
from .tuning import (
    FALLBACK_HYPERPARAMETERS,
    hyperparameters_to_dict,
    tune_tfidf,
)
from .vector_store import TfidfIndexMetadata, TfidfVectorStore

INDEX_REQUIRED_FILES = (
    "vectorizer.joblib",
    "doc_matrix.npz",
    "documents.jsonl",
    "config.json",
)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    chunks.append(str(item["text"]))
                else:
                    chunks.append(json.dumps(item, ensure_ascii=False))
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    if content is None:
        return ""
    return str(content).strip()


def _extract_steps(answer_text: str) -> list[str]:
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    numbered = []
    for line in lines:
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned and cleaned != line or re.match(r"^\d+[.)]\s+", line):
            numbered.append(cleaned)
    if numbered:
        return numbered[:5]

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", answer_text) if segment.strip()]
    return sentences[:5]


def _messages_from_result(result: Any) -> list[Any]:
    if isinstance(result, dict):
        return list(result.get("messages", []))
    return []


def _used_retrieval_tool(messages: list[Any]) -> bool:
    return any(isinstance(message, ToolMessage) for message in messages)


def _last_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _content_to_text(message.content)
            if text:
                return text
    return ""


def _build_system_prompt(*, call_emergency_now: bool, stricter: bool) -> str:
    prompt = (
        "You are a first-aid co-pilot. "
        "You must call the tool search_first_aid_knowledge before answering any user question. "
        "Only use information grounded in the retrieved first-aid guidance. "
        "Do not invent steps. "
        "Answer in concise numbered steps."
    )
    if call_emergency_now:
        prompt += " State the need to contact emergency services in the first one or two steps when appropriate."
    if stricter:
        prompt += (
            " This is a retry. You must use the retrieval tool, stay source-grounded, "
            "and explicitly include emergency escalation language when required."
        )
    return prompt


class FirstAidCopilotService:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.config.ensure_runtime_dirs()
        self.logger = ConversationLogger(self.config.conversations_dir)
        self._store_cache: dict[str, TfidfVectorStore] = {}

    def _index_built(self, profile: ProfileName) -> bool:
        index_dir = self.config.index_dir(profile)
        return all((index_dir / name).exists() for name in INDEX_REQUIRED_FILES)

    def _load_store(self, profile: ProfileName) -> TfidfVectorStore:
        if profile in self._store_cache:
            return self._store_cache[profile]
        if not self._index_built(profile):
            raise FileNotFoundError(
                f"Index for profile '{profile}' is missing. Run build-index first."
            )
        store = TfidfVectorStore.load(self.config.index_dir(profile))
        self._store_cache[profile] = store
        return store

    def _ollama_models(self) -> tuple[bool, set[str]]:
        url = self.config.ollama_base_url.rstrip("/") + "/api/tags"
        try:
            response = httpx.get(url, timeout=5.0)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return False, set()

        models = {
            str(model.get("name", ""))
            for model in payload.get("models", [])
            if model.get("name")
        }
        return True, models

    def _resolve_runtime_model_name(self, requested_model: str) -> str:
        self.config.validate_model(requested_model)
        available, exact_models = self._ollama_models()
        if not available:
            return requested_model
        if requested_model in exact_models:
            return requested_model

        requested_normalized = normalize_model_name(requested_model)
        for available_model in exact_models:
            if normalize_model_name(available_model) == requested_normalized:
                return available_model
        return requested_model

    def model_statuses(self) -> list[ModelStatus]:
        available, models = self._ollama_models()
        normalized_models = {normalize_model_name(name) for name in models}
        return [
            ModelStatus(
                name=name,
                available=available and normalize_model_name(name) in normalized_models,
            )
            for name in self.config.model_names
        ]

    def doctor(self) -> DoctorReport:
        return DoctorReport(
            python_executable=str(Path(__import__("sys").executable)),
            venv_exists=(self.config.root_dir / ".venv").exists(),
            ollama_available=self._ollama_models()[0],
            models=self.model_statuses(),
            indexes_built={
                profile: self._index_built(profile)
                for profile in ("experiment", "demo")
            },
        )

    def health_status(self) -> HealthResponse:
        ollama_available, _models = self._ollama_models()
        return HealthResponse(
            status="ok",
            ollama_available=ollama_available,
            available_profiles=[
                profile for profile in ("experiment", "demo") if self._index_built(profile)
            ],
            configured_models=self.model_statuses(),
        )

    def build_index(self, profile: str, *, force: bool = False) -> Path:
        validated_profile = self.config.validate_profile(profile)
        index_dir = self.config.index_dir(validated_profile)
        if self._index_built(validated_profile) and not force:
            return index_dir

        source_split = get_profile_source_split(validated_profile)
        train_frame = load_split_dataframe(self.config, "train")
        dev_frame = load_split_dataframe(self.config, "dev")
        train_documents = build_documents(train_frame, "train")

        try:
            tuning_result = tune_tfidf(
                train_texts=[document.page_content for document in train_documents],
                train_answers=[str(answer) for answer in train_frame["answer"].tolist()],
                train_categories=[str(category) for category in train_frame["category"].tolist()],
                train_doc_ids=[str(document.metadata["doc_id"]) for document in train_documents],
                dev_queries=[str(query) for query in dev_frame["question"].tolist()],
                dev_answers=[str(answer) for answer in dev_frame["answer"].tolist()],
                dev_categories=[str(category) for category in dev_frame["category"].tolist()],
            )
            chosen_params = tuning_result.best_params
            tuning_payload = {
                "best_score": tuning_result.best_score,
                "candidate_count": tuning_result.candidate_count,
                "best_metrics": tuning_result.best_metrics,
                "best_params": hyperparameters_to_dict(tuning_result.best_params),
            }
        except Exception as exc:
            chosen_params = FALLBACK_HYPERPARAMETERS
            tuning_payload = {
                "best_score": None,
                "candidate_count": 0,
                "best_metrics": {},
                "best_params": hyperparameters_to_dict(FALLBACK_HYPERPARAMETERS),
                "error": str(exc),
            }

        if validated_profile == "experiment":
            documents = train_documents
        else:
            demo_frame = load_split_dataframe(self.config, source_split)
            documents = build_documents(demo_frame, source_split)

        metadata = TfidfIndexMetadata(
            profile=validated_profile,
            source_split=source_split,
            hyperparameters=chosen_params,
            tuning=tuning_payload,
        )
        store = TfidfVectorStore.build(documents, metadata)
        store.save(index_dir)
        self._store_cache[validated_profile] = store
        return index_dir

    def retrieve(self, query: str, profile: str, *, k: int) -> list[RetrievalHit]:
        validated_profile = self.config.validate_profile(profile)
        store = self._load_store(validated_profile)
        hits = store.similarity_search_with_scores(query, k=k)
        return [
            RetrievalHit(
                doc_id=str(document.metadata["doc_id"]),
                question=str(document.metadata["question"]),
                answer=str(document.metadata["answer"]),
                category=str(document.metadata["category"]),
                source=str(document.metadata["source"]),
                split=str(document.metadata["split"]),
                score=round(score, 6),
            )
            for document, score in hits
        ]

    def _build_model(self, model_name: str) -> ChatOllama:
        runtime_model_name = self._resolve_runtime_model_name(model_name)
        return ChatOllama(
            model=runtime_model_name,
            base_url=self.config.ollama_base_url,
            temperature=0.1,
            validate_model_on_init=False,
        )

    def _run_agent(
        self,
        query: str,
        *,
        model_name: str,
        profile: ProfileName,
        top_k: int,
        call_emergency_now: bool,
        stricter: bool,
    ) -> tuple[str, list[RetrievalHit], bool]:
        tool_state: dict[str, Any] = {"hits": []}

        @tool("search_first_aid_knowledge")
        def search_first_aid_knowledge(query: str, k: int = 3) -> str:
            """Search the first-aid knowledge base and return the closest matching guidance."""
            safe_k = max(1, min(k, self.config.default_debug_top_k))
            retrieval_hits = [
                hit.model_dump()
                for hit in self.retrieve(query, profile, k=safe_k)
            ]
            tool_state["hits"] = retrieval_hits
            return json.dumps(retrieval_hits, ensure_ascii=False)

        agent = create_agent(
            model=self._build_model(model_name),
            tools=[search_first_aid_knowledge],
            system_prompt=_build_system_prompt(
                call_emergency_now=call_emergency_now,
                stricter=stricter,
            ),
            name=f"firstaid-{model_name}-{profile}",
        )
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = _messages_from_result(result)
        final_text = _last_ai_text(messages)
        used_tool = _used_retrieval_tool(messages)
        raw_hits = tool_state["hits"]
        retrieval_hits = [RetrievalHit(**payload) for payload in raw_hits]
        if not retrieval_hits:
            retrieval_hits = self.retrieve(query, profile, k=top_k)
        return final_text, retrieval_hits, used_tool

    def _fallback_answer(self, query: str, retrieval_hits: list[RetrievalHit], call_emergency_now: bool) -> str:
        if not retrieval_hits:
            if call_emergency_now:
                return (
                    "Call emergency services now. I could not retrieve grounded guidance reliably enough "
                    "to provide a safe answer."
                )
            return "I could not retrieve grounded first-aid guidance for that question."

        best_hit = retrieval_hits[0]
        prefix = ""
        if call_emergency_now:
            prefix = "Call emergency services now if the person is in immediate danger, not breathing, or worsening.\n\n"
        return (
            f"{prefix}Closest matching first-aid guidance:\n"
            f"{best_hit.answer}"
        )

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        profile = self.config.validate_profile(request.profile)
        self.config.validate_model(request.model)
        safety = assess_query(request.query)
        session_id = sanitize_identifier(request.session_id) if request.session_id else make_session_id()
        turn_id = make_turn_id()
        warnings = list(safety.warnings)

        try:
            answer_text, retrieval_hits, used_tool = self._run_agent(
                request.query,
                model_name=request.model,
                profile=profile,
                top_k=request.top_k,
                call_emergency_now=safety.call_emergency_now,
                stricter=False,
            )
        except Exception as exc:
            retrieval_hits = self.retrieve(request.query, profile, k=request.top_k)
            answer_text = self._fallback_answer(
                request.query,
                retrieval_hits,
                safety.call_emergency_now,
            )
            used_tool = False
            warnings.append(f"Agent invocation failed; returned retrieval-backed fallback. Error: {exc}")

        needs_retry = (not used_tool) or (
            safety.call_emergency_now and not has_required_emergency_language(answer_text)
        )

        if needs_retry:
            try:
                retried_answer, retried_hits, retried_used_tool = self._run_agent(
                    request.query,
                    model_name=request.model,
                    profile=profile,
                    top_k=request.top_k,
                    call_emergency_now=safety.call_emergency_now,
                    stricter=True,
                )
                answer_text = retried_answer or answer_text
                retrieval_hits = retried_hits or retrieval_hits
                used_tool = retried_used_tool or used_tool
            except Exception as exc:
                warnings.append(f"Retry failed: {exc}")

        if safety.call_emergency_now and not has_required_emergency_language(answer_text):
            warnings.append("Emergency escalation language was injected outside the model response.")
            answer_text = (
                "Call emergency services now if the person is not breathing, losing consciousness, "
                "or worsening.\n\n" + answer_text
            ).strip()

        if not answer_text.strip():
            answer_text = self._fallback_answer(
                request.query,
                retrieval_hits,
                safety.call_emergency_now,
            )
            warnings.append("Agent returned an empty answer; used retrieval-backed fallback.")

        steps = _extract_steps(answer_text)
        if not used_tool:
            warnings.append("The retrieval tool was not used on the first attempt.")

        response = QueryResponse(
            session_id=session_id,
            turn_id=turn_id,
            query=request.query,
            model=request.model,
            profile=profile,
            risk_category=safety.risk_category,
            call_emergency_now=safety.call_emergency_now,
            steps=steps,
            answer_text=answer_text,
            sources=retrieval_hits,
            retrieval_hits=retrieval_hits,
            warnings=warnings,
            used_retrieval_tool=used_tool,
        )

        log_payload = {
            "session_id": session_id,
            "turn_id": turn_id,
            "timestamp": utc_now_iso(),
            "user_query": request.query,
            "model": request.model,
            "profile": profile,
            "risk_category": safety.risk_category,
            "retrieval_hits": [hit.model_dump() for hit in retrieval_hits],
            "final_answer": answer_text,
            "warnings": warnings,
        }
        self.logger.log_turn(session_id, log_payload)
        self.logger.log_run(
            {
                "timestamp": utc_now_iso(),
                "session_id": session_id,
                "turn_id": turn_id,
                "model": request.model,
                "profile": profile,
                "used_retrieval_tool": used_tool,
            }
        )
        return response
