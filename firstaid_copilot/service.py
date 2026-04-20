from __future__ import annotations

# Core orchestration: retrieval, local model calls, safety retries, and logging.

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama

from .config import AppConfig, ProfileName, normalize_model_name
from .conversation import (
    ConversationLogger,
    trace_message_from_langchain,
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
    ConversationSummary,
    ConversationTraceMessage,
    ConversationThread,
    DoctorReport,
    HealthResponse,
    ModelStatus,
    QueryRequest,
    QueryResponse,
    RetrievalHit,
    StreamEvent,
)
from .tuning import (
    FALLBACK_HYPERPARAMETERS,
    hyperparameters_to_dict,
    select_tfidf_with_dev_test,
    selection_result_to_dict,
    TfidfSelectionResult,
)
from .vector_store import TfidfIndexMetadata, TfidfVectorStore

INDEX_REQUIRED_FILES = (
    "vectorizer.joblib",
    "doc_matrix.npz",
    "documents.jsonl",
    "config.json",
)
MAX_HISTORY_HUMAN_MESSAGES = 5
TFIDF_EVALUATION_FILENAME = "tfidf_selection.json"


def _content_to_text(content: Any) -> str:
    """Convert model message content into stripped plain text."""
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


def _content_to_chunk_text(content: Any) -> str:
    """Convert streamed message content into raw chunk text."""
    if isinstance(content, str):
        return content
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
        return "".join(chunks)
    if content is None:
        return ""
    return str(content)


def _strip_tool_markup(answer_text: str, *, trim: bool = True) -> str:
    """Remove accidental tool-call markup from answer text."""
    # Some local models emit tool XML as plain text; remove it before display/logging.
    cleaned = re.sub(
        r"<search_first_aid_knowledge>.*?</search_first_aid_knowledge>",
        "",
        answer_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"</?search_first_aid_knowledge>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip() if trim else cleaned


def _extract_steps(answer_text: str) -> list[str]:
    """Extract up to five concise answer steps."""
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    numbered = []
    for line in lines:
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if (cleaned and cleaned != line) or re.match(r"^\d+[.)]\s+", line):
            numbered.append(cleaned)
    if numbered:
        return numbered[:5]

    sentences = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+", answer_text)
        if segment.strip()
    ]
    return sentences[:5]


def _messages_from_result(result: Any) -> list[Any]:
    """Extract LangChain messages from an agent result."""
    if isinstance(result, dict):
        return list(result.get("messages", []))
    return []


def _used_retrieval_tool(messages: list[Any]) -> bool:
    """Return whether a tool message appeared in agent messages."""
    return any(isinstance(message, ToolMessage) for message in messages)


def _last_ai_text(messages: list[Any]) -> str:
    """Return the most recent non-empty assistant text."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _strip_tool_markup(_content_to_text(message.content))
            if text:
                return text
    return ""


def _build_agent_input(
    history_messages: list[BaseMessage],
    query: str,
) -> dict[str, Any]:
    """Build the message payload passed into the LangChain agent."""
    return {"messages": [*history_messages, HumanMessage(content=query)]}


def _build_system_prompt(
    *,
    call_emergency_now: bool,
    stricter: bool,
) -> str:
    """Build the retrieval-first system prompt for the local agent."""
    prompt = (
        """You are a helpful medical assistant continuing an ongoing conversation.

Available tool:
- search_first_aid_knowledge: Use this tool to search for first-aid knowledge when the user is asking for factual guidance or when the conversation context is insufficient.

General Operating Principles:
- Use the prior conversation context when it is relevant to the user's latest message.
- If the current turn is a conversational follow-up and the conversation context is already sufficient, answer directly without forcing a tool call.
- If the user is asking for first-aid instructions, medical facts, or the conversation context is not enough, call search_first_aid_knowledge before answering.
- When you use retrieved information, base your factual guidance on it and do not fabricate unsupported details.
- You may call the tool multiple times if needed.

Response Expectations:
- Provide a clear and direct answer to the user's latest message.
- If you do not have enough information to provide a safe answer, say so clearly and advise emergency services if the situation is urgent or life-threatening.
- If you use retrieved evidence, give one unified answer rather than separate answers per source.

Escalation and Limits:
- If the user's query indicates a potentially life-threatening situation or an emergency, explicitly state that they should call emergency services immediately. Mention this as early as possible in your answer.
- Do not fabricate information."""
    )
    if call_emergency_now:
        prompt += (
            "\n- State the need to contact emergency services in the first one or two "
            "steps when appropriate."
        )
    if stricter:
        prompt += (
            "\n\nIMPORTANT:\n- Your previous answer was empty, incomplete, or unsafe."
            "\n- Use the conversation context carefully."
            "\n- If you need grounded first-aid guidance, call search_first_aid_knowledge."
            "\n- Return a complete, direct answer to the user's latest message."
        )
    return prompt


def _normalize_agent_stream_part(part: Any) -> dict[str, Any]:
    """Normalize LangChain stream chunks into a consistent dictionary shape."""
    if isinstance(part, dict):
        return part
    if isinstance(part, tuple):
        if len(part) == 2:
            mode, payload = part
            return {"type": mode, "ns": (), "data": payload}
        if len(part) == 3:
            namespace, mode, payload = part
            return {"type": mode, "ns": namespace, "data": payload}
    raise TypeError(f"Unsupported stream part shape: {type(part)!r}")


def _append_trace_message(
    trace_messages: list[ConversationTraceMessage],
    message: BaseMessage,
) -> None:
    """Append a non-duplicate trace message when it can be converted."""
    trace_message = trace_message_from_langchain(message)
    if trace_message is None:
        return
    if trace_messages and trace_messages[-1] == trace_message:
        return
    trace_messages.append(trace_message)


def _build_turn_trace_messages(
    query: str,
    *,
    new_messages: list[BaseMessage],
    answer_text: str,
) -> list[ConversationTraceMessage]:
    """Build trace messages for one non-streaming agent turn."""
    trace_messages: list[ConversationTraceMessage] = [
        ConversationTraceMessage(role="human", content=query)
    ]
    for message in new_messages:
        if isinstance(message, HumanMessage):
            continue
        _append_trace_message(trace_messages, message)

    final_answer = _strip_tool_markup(answer_text)
    if final_answer:
        final_trace = ConversationTraceMessage(
            role="assistant",
            content=final_answer,
        )
        if not trace_messages or trace_messages[-1] != final_trace:
            trace_messages.append(final_trace)
    return trace_messages


def _compose_final_turn_trace_messages(
    query: str,
    *,
    trace_messages: list[ConversationTraceMessage],
    answer_text: str,
) -> list[ConversationTraceMessage]:
    """Finalize trace messages with one clean assistant response."""
    finalized: list[ConversationTraceMessage] = []
    for trace_message in trace_messages:
        if trace_message.role == "assistant" and not trace_message.tool_calls:
            continue
        if not finalized or finalized[-1] != trace_message:
            finalized.append(trace_message)

    if not finalized or finalized[0].role != "human":
        finalized.insert(0, ConversationTraceMessage(role="human", content=query))

    clean_answer = _strip_tool_markup(answer_text)
    if clean_answer:
        finalized.append(
            ConversationTraceMessage(
                role="assistant",
                content=clean_answer,
            )
        )
    return finalized


def _iter_message_objects(value: Any) -> list[BaseMessage]:
    """Recursively collect LangChain message objects from stream payloads."""
    messages: list[BaseMessage] = []
    if isinstance(value, BaseMessage):
        messages.append(value)
        return messages
    if isinstance(value, dict):
        for item in value.values():
            messages.extend(_iter_message_objects(item))
        return messages
    if isinstance(value, (list, tuple)):
        for item in value:
            messages.extend(_iter_message_objects(item))
        return messages
    return messages


@dataclass(slots=True)
class AgentAttemptResult:
    answer_text: str = ""
    retrieval_hits: list[RetrievalHit] = field(default_factory=list)
    used_tool: bool = False
    trace_messages: list[ConversationTraceMessage] = field(default_factory=list)


class FirstAidCopilotService:
    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialize service dependencies and runtime directories."""
        self.config = config or AppConfig()
        self.config.ensure_runtime_dirs()
        self.logger = ConversationLogger(self.config.conversations_dir)
        self._store_cache: dict[str, TfidfVectorStore] = {}

    def _index_built(self, profile: ProfileName) -> bool:
        """Return whether all required index files exist."""
        index_dir = self.config.index_dir(profile)
        return all((index_dir / name).exists() for name in INDEX_REQUIRED_FILES)

    def _load_store(self, profile: ProfileName) -> TfidfVectorStore:
        """Load or reuse the vector store for a profile."""
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
        """Fetch locally available Ollama model names."""
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
        """Resolve aliases such as functiongemma:latest for Ollama calls."""
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
        """Return availability for all configured models."""
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
        """Return a diagnostic report for CLI doctor output."""
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
        """Return API health and availability status."""
        ollama_available, _models = self._ollama_models()
        return HealthResponse(
            status="ok",
            ollama_available=ollama_available,
            available_profiles=[
                profile
                for profile in ("experiment", "demo")
                if self._index_built(profile)
            ],
            configured_models=self.model_statuses(),
        )

    def list_conversations(self) -> list[ConversationSummary]:
        """Return persisted conversation summaries."""
        return self.logger.list_conversations()

    def get_conversation_thread(self, session_id: str) -> ConversationThread:
        """Return a persisted conversation thread."""
        return self.logger.load_conversation(session_id)

    def _run_tfidf_selection(
        self,
        *,
        train_frame,
        dev_frame,
        test_frame,
        train_documents: list[Any],
    ) -> TfidfSelectionResult:
        """Run TF-IDF selection using train, dev, and test frames."""
        return select_tfidf_with_dev_test(
            train_texts=[document.page_content for document in train_documents],
            train_answers=[str(answer) for answer in train_frame["answer"].tolist()],
            train_categories=[
                str(category) for category in train_frame["category"].tolist()
            ],
            dev_queries=[str(query) for query in dev_frame["question"].tolist()],
            dev_answers=[str(answer) for answer in dev_frame["answer"].tolist()],
            dev_categories=[
                str(category) for category in dev_frame["category"].tolist()
            ],
            test_queries=[str(query) for query in test_frame["question"].tolist()],
            test_answers=[str(answer) for answer in test_frame["answer"].tolist()],
            test_categories=[
                str(category) for category in test_frame["category"].tolist()
            ],
        )

    def _save_tfidf_selection(self, selection_result: TfidfSelectionResult) -> Path:
        """Persist the TF-IDF selection result as JSON."""
        self.config.evaluations_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.evaluations_dir / TFIDF_EVALUATION_FILENAME
        with path.open("w", encoding="utf-8") as handle:
            json.dump(selection_result_to_dict(selection_result), handle, indent=2)
        return path

    def evaluate_tfidf(self) -> tuple[TfidfSelectionResult, Path]:
        """Evaluate and save the selected TF-IDF configuration."""
        train_frame = load_split_dataframe(self.config, "train")
        dev_frame = load_split_dataframe(self.config, "dev")
        test_frame = load_split_dataframe(self.config, "test")
        train_documents = build_documents(train_frame, "train")
        selection_result = self._run_tfidf_selection(
            train_frame=train_frame,
            dev_frame=dev_frame,
            test_frame=test_frame,
            train_documents=train_documents,
        )
        return selection_result, self._save_tfidf_selection(selection_result)

    def build_index(self, profile: str, *, force: bool = False) -> Path:
        """Build and persist a retrieval index for a profile."""
        validated_profile = self.config.validate_profile(profile)
        index_dir = self.config.index_dir(validated_profile)
        if self._index_built(validated_profile) and not force:
            return index_dir

        source_split = get_profile_source_split(validated_profile)
        train_frame = load_split_dataframe(self.config, "train")
        dev_frame = load_split_dataframe(self.config, "dev")
        test_frame = load_split_dataframe(self.config, "test")
        train_documents = build_documents(train_frame, "train")

        try:
            selection_result = self._run_tfidf_selection(
                train_frame=train_frame,
                dev_frame=dev_frame,
                test_frame=test_frame,
                train_documents=train_documents,
            )
            self._save_tfidf_selection(selection_result)
            chosen_params = selection_result.best_params
            tuning_payload = selection_result_to_dict(selection_result)
        except Exception as exc:
            chosen_params = FALLBACK_HYPERPARAMETERS
            tuning_payload = {
                "selection_strategy": "top_dev_then_weighted_dev_test",
                "best_weighted_score": None,
                "candidate_count": 0,
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
        """Retrieve top-k first-aid documents for a query."""
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
        """Create the ChatOllama wrapper for a configured model."""
        runtime_model_name = self._resolve_runtime_model_name(model_name)
        return ChatOllama(
            model=runtime_model_name,
            base_url=self.config.ollama_base_url,
            temperature=0.1,
            sync_client_kwargs={"timeout": self.config.request_timeout_seconds},
            async_client_kwargs={"timeout": self.config.request_timeout_seconds},
            validate_model_on_init=False,
        )

    def _stream_event(self, event_type: str, **data: Any) -> StreamEvent:
        """Create a typed stream event."""
        return StreamEvent(type=event_type, data=data)

    def _coerce_retrieval_hits(self, raw_hits: list[dict[str, Any]]) -> list[RetrievalHit]:
        """Convert raw retrieval payloads into RetrievalHit models."""
        return [RetrievalHit(**payload) for payload in raw_hits]

    def _history_messages(self, session_id: str) -> list[BaseMessage]:
        """Load bounded conversation history for an agent call."""
        return self.logger.load_context_messages(
            session_id,
            max_human_messages=MAX_HISTORY_HUMAN_MESSAGES,
        )

    def _create_agent_runner(
        self,
        *,
        model_name: str,
        profile: ProfileName,
        top_k: int,
        call_emergency_now: bool,
        stricter: bool,
    ) -> tuple[Any, dict[str, Any]]:
        """Create a LangChain agent and capture state for retrieval hits."""
        tool_state: dict[str, Any] = {"hits": []}

        @tool("search_first_aid_knowledge")
        def search_first_aid_knowledge(query: str, k: int = top_k) -> str:
            """Search the medical knowledge base based on the search query."""
            # Bound model-provided k so tool calls stay within the app's debug limit.
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
        return agent, tool_state

    def _run_agent(
        self,
        query: str,
        *,
        model_name: str,
        profile: ProfileName,
        top_k: int,
        call_emergency_now: bool,
        stricter: bool,
        history_messages: list[BaseMessage],
    ) -> AgentAttemptResult:
        """Run one synchronous agent attempt."""
        agent, tool_state = self._create_agent_runner(
            model_name=model_name,
            profile=profile,
            top_k=top_k,
            call_emergency_now=call_emergency_now,
            stricter=stricter,
        )
        result = agent.invoke(_build_agent_input(history_messages, query))
        messages = _messages_from_result(result)
        history_count = len(history_messages)
        new_messages = (
            list(messages[history_count:])
            if len(messages) >= history_count
            else list(messages)
        )
        attempt_messages = [
            message
            for message in new_messages
            if isinstance(message, BaseMessage)
        ]
        final_text = _strip_tool_markup(_last_ai_text(attempt_messages))
        return AgentAttemptResult(
            answer_text=final_text,
            retrieval_hits=self._coerce_retrieval_hits(tool_state["hits"]),
            used_tool=_used_retrieval_tool(attempt_messages),
            trace_messages=_build_turn_trace_messages(
                query,
                new_messages=attempt_messages,
                answer_text=final_text,
            ),
        )

    async def _emit_agent_attempt(
        self,
        query: str,
        *,
        model_name: str,
        profile: ProfileName,
        top_k: int,
        call_emergency_now: bool,
        stricter: bool,
        history_messages: list[BaseMessage],
        attempt_state: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        """Stream one agent attempt while collecting answer and retrieval state."""
        attempt_state.clear()
        attempt_state.update(
            {
                "answer_text": "",
                "retrieval_hits": [],
                "used_tool": False,
                "trace_messages": [ConversationTraceMessage(role="human", content=query)],
                "streamed_text": "",
                "streamed_tokens": False,
            }
        )

        agent, tool_state = self._create_agent_runner(
            model_name=model_name,
            profile=profile,
            top_k=top_k,
            call_emergency_now=call_emergency_now,
            stricter=stricter,
        )

        answer_parts: list[str] = []
        retrieval_status_emitted = False
        emitted_retrieval_signatures: set[str] = set()
        observed_trace_messages: list[ConversationTraceMessage] = list(
            attempt_state["trace_messages"]
        )

        async for part in agent.astream(
            _build_agent_input(history_messages, query),
            stream_mode=["messages", "updates"],
        ):
            normalized_part = _normalize_agent_stream_part(part)
            part_type = normalized_part.get("type")
            if part_type == "updates":
                updates = normalized_part.get("data", {})
                for observed_message in _iter_message_objects(updates):
                    if isinstance(observed_message, ToolMessage):
                        attempt_state["used_tool"] = True
                        _append_trace_message(observed_trace_messages, observed_message)
                    elif isinstance(observed_message, AIMessage) and observed_message.tool_calls:
                        _append_trace_message(observed_trace_messages, observed_message)

                candidate_text = _strip_tool_markup(
                    _last_ai_text(_iter_message_objects(updates))
                )
                if candidate_text:
                    attempt_state["answer_text"] = candidate_text

                raw_hits = tool_state["hits"]
                if raw_hits:
                    current_hits = self._coerce_retrieval_hits(raw_hits)
                    attempt_state["retrieval_hits"] = current_hits
                    signature = json.dumps(
                        [hit.model_dump(mode="json") for hit in current_hits],
                        sort_keys=True,
                    )
                    if signature not in emitted_retrieval_signatures:
                        emitted_retrieval_signatures.add(signature)
                        if not retrieval_status_emitted:
                            yield self._stream_event("status", value="retrieving")
                            retrieval_status_emitted = True
                        yield self._stream_event(
                            "retrieval",
                            hits=[
                                hit.model_dump(mode="json") for hit in current_hits
                            ],
                        )
                continue

            if part_type != "messages":
                continue

            message, _metadata = normalized_part.get("data", (None, {}))
            if isinstance(message, ToolMessage):
                attempt_state["used_tool"] = True
                _append_trace_message(observed_trace_messages, message)
                continue
            if isinstance(message, AIMessage) and message.tool_calls:
                _append_trace_message(observed_trace_messages, message)
                continue
            if not isinstance(message, AIMessageChunk):
                continue

            chunk_text = _strip_tool_markup(
                _content_to_chunk_text(message.content),
                trim=False,
            )
            if not chunk_text:
                continue

            answer_parts.append(chunk_text)
            attempt_state["streamed_tokens"] = True
            attempt_state["streamed_text"] = "".join(answer_parts)
            yield self._stream_event("token", text=chunk_text)

        streamed_text = _strip_tool_markup("".join(answer_parts))
        if streamed_text.strip():
            attempt_state["answer_text"] = streamed_text.strip()
        raw_hits = tool_state["hits"]
        if raw_hits:
            attempt_state["retrieval_hits"] = self._coerce_retrieval_hits(raw_hits)
        attempt_state["trace_messages"] = _compose_final_turn_trace_messages(
            query,
            trace_messages=observed_trace_messages,
            answer_text=str(attempt_state["answer_text"]),
        )

    def _fallback_answer(
        self,
        query: str,
        retrieval_hits: list[RetrievalHit],
        call_emergency_now: bool,
    ) -> str:
        """Build a retrieval-backed answer when generation fails."""
        if not retrieval_hits:
            if call_emergency_now:
                return (
                    "Call emergency services now. I could not retrieve grounded "
                    "guidance reliably enough to provide a safe answer."
                )
            return "I could not retrieve grounded first-aid guidance for that question."

        best_hit = retrieval_hits[0]
        prefix = ""
        if call_emergency_now:
            prefix = (
                "Call emergency services now if the person is in immediate danger, "
                "not breathing, or worsening.\n\n"
            )
        return f"{prefix}Closest matching first-aid guidance:\n{best_hit.answer}"

    def _build_response(
        self,
        request: QueryRequest,
        *,
        session_id: str,
        turn_id: str,
        risk_category: str,
        call_emergency_now: bool,
        answer_text: str,
        retrieval_hits: list[RetrievalHit],
        warnings: list[str],
        used_retrieval_tool: bool,
    ) -> QueryResponse:
        """Create the public query response model."""
        clean_answer = _strip_tool_markup(answer_text).strip()
        return QueryResponse(
            session_id=session_id,
            turn_id=turn_id,
            query=request.query,
            model=request.model,
            profile=request.profile,
            risk_category=risk_category,
            call_emergency_now=call_emergency_now,
            steps=_extract_steps(clean_answer),
            answer_text=clean_answer,
            sources=retrieval_hits,
            retrieval_hits=retrieval_hits,
            warnings=warnings,
            used_retrieval_tool=used_retrieval_tool,
        )

    def _log_response(
        self,
        response: QueryResponse,
        *,
        trace_messages: list[ConversationTraceMessage],
    ) -> None:
        """Persist the completed turn and compact run metadata."""
        log_payload = {
            "session_id": response.session_id,
            "turn_id": response.turn_id,
            "timestamp": utc_now_iso(),
            "user_query": response.query,
            "model": response.model,
            "profile": response.profile,
            "risk_category": response.risk_category,
            "retrieval_hits": [hit.model_dump() for hit in response.retrieval_hits],
            "final_answer": response.answer_text,
            "warnings": response.warnings,
            "trace_messages": [
                message.model_dump(mode="json")
                for message in trace_messages
            ],
        }
        self.logger.log_turn(response.session_id, log_payload)
        self.logger.log_run(
            {
                "timestamp": utc_now_iso(),
                "session_id": response.session_id,
                "turn_id": response.turn_id,
                "model": response.model,
                "profile": response.profile,
                "used_retrieval_tool": response.used_retrieval_tool,
            }
        )

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        """Answer one query with retrieval, generation, retry, and fallback logic."""
        profile = self.config.validate_profile(request.profile)
        self.config.validate_model(request.model)
        safety = assess_query(request.query)
        session_id = (
            sanitize_identifier(request.session_id)
            if request.session_id
            else make_session_id()
        )
        history_messages = self._history_messages(session_id)
        turn_id = make_turn_id()
        warnings = list(safety.warnings)

        first_attempt = AgentAttemptResult()
        first_attempt_error: Exception | None = None

        try:
            first_attempt = self._run_agent(
                request.query,
                model_name=request.model,
                profile=profile,
                top_k=request.top_k,
                call_emergency_now=safety.call_emergency_now,
                stricter=False,
                history_messages=history_messages,
            )
        except Exception as exc:
            first_attempt_error = exc
            warnings.append(f"Agent invocation failed on the first attempt. Error: {exc}")

        final_attempt = first_attempt
        overall_used_tool = first_attempt.used_tool

        # Retry only for failed/empty/unsafe answers, not just because no tool was used.
        needs_retry = (first_attempt_error is not None) or (not first_attempt.answer_text.strip()) or (
            safety.call_emergency_now
            and not has_required_emergency_language(first_attempt.answer_text)
        )

        if needs_retry:
            try:
                retry_attempt = self._run_agent(
                    request.query,
                    model_name=request.model,
                    profile=profile,
                    top_k=request.top_k,
                    call_emergency_now=safety.call_emergency_now,
                    stricter=True,
                    history_messages=history_messages,
                )
                overall_used_tool = retry_attempt.used_tool or overall_used_tool
                if retry_attempt.answer_text.strip():
                    final_attempt = retry_attempt
                else:
                    final_attempt = AgentAttemptResult(
                        answer_text=final_attempt.answer_text,
                        retrieval_hits=retry_attempt.retrieval_hits or final_attempt.retrieval_hits,
                        used_tool=overall_used_tool,
                        trace_messages=retry_attempt.trace_messages or final_attempt.trace_messages,
                    )
            except Exception as exc:
                warnings.append(f"Retry failed: {exc}")

        answer_text = final_attempt.answer_text
        retrieval_hits = list(final_attempt.retrieval_hits)
        trace_messages = list(final_attempt.trace_messages)

        if not answer_text.strip():
            if not retrieval_hits:
                retrieval_hits = self.retrieve(request.query, profile, k=request.top_k)
            answer_text = self._fallback_answer(
                request.query,
                retrieval_hits,
                safety.call_emergency_now,
            )
            warnings.append(
                "Agent returned an empty answer; used retrieval-backed fallback."
            )
            trace_messages = _compose_final_turn_trace_messages(
                request.query,
                trace_messages=[ConversationTraceMessage(role="human", content=request.query)],
                answer_text=answer_text,
            )

        if safety.call_emergency_now and not has_required_emergency_language(answer_text):
            warnings.append(
                "Emergency escalation language was injected outside the model response."
            )
            answer_text = (
                "Call emergency services now if the person is not breathing, "
                "losing consciousness, or worsening.\n\n" + answer_text
            ).strip()
        trace_messages = _compose_final_turn_trace_messages(
            request.query,
            trace_messages=trace_messages or [ConversationTraceMessage(role="human", content=request.query)],
            answer_text=answer_text,
        )

        response = self._build_response(
            request,
            session_id=session_id,
            turn_id=turn_id,
            risk_category=safety.risk_category,
            call_emergency_now=safety.call_emergency_now,
            answer_text=answer_text,
            retrieval_hits=retrieval_hits,
            warnings=warnings,
            used_retrieval_tool=overall_used_tool,
        )
        self._log_response(response, trace_messages=trace_messages)
        return response

    async def astream_query(self, request: QueryRequest) -> AsyncIterator[StreamEvent]:
        """Stream a query response through status, token, warning, and final events."""
        profile = self.config.validate_profile(request.profile)
        self.config.validate_model(request.model)
        safety = assess_query(request.query)
        session_id = (
            sanitize_identifier(request.session_id)
            if request.session_id
            else make_session_id()
        )
        history_messages = self._history_messages(session_id)
        turn_id = make_turn_id()
        warnings = list(safety.warnings)

        yield self._stream_event(
            "session",
            session_id=session_id,
            turn_id=turn_id,
            model=request.model,
            profile=profile,
            risk_category=safety.risk_category,
            call_emergency_now=safety.call_emergency_now,
        )
        yield self._stream_event("status", value="started")

        first_attempt_state: dict[str, Any] = {}
        first_attempt_error: Exception | None = None

        try:
            async for event in self._emit_agent_attempt(
                request.query,
                model_name=request.model,
                profile=profile,
                top_k=request.top_k,
                call_emergency_now=safety.call_emergency_now,
                stricter=False,
                history_messages=history_messages,
                attempt_state=first_attempt_state,
            ):
                yield event
        except Exception as exc:
            first_attempt_error = exc
            warnings.append(f"Agent invocation failed on the first attempt. Error: {exc}")
            first_attempt_state = {
                "answer_text": "",
                "retrieval_hits": [],
                "used_tool": False,
                "trace_messages": [],
                "streamed_text": "",
                "streamed_tokens": False,
            }

        first_attempt = AgentAttemptResult(
            answer_text=str(first_attempt_state.get("answer_text", "")),
            retrieval_hits=list(first_attempt_state.get("retrieval_hits", [])),
            used_tool=bool(first_attempt_state.get("used_tool", False)),
            trace_messages=list(first_attempt_state.get("trace_messages", [])),
        )
        final_attempt = first_attempt
        overall_used_tool = first_attempt.used_tool

        # Streaming follows the same retry rules as the non-streaming path.
        needs_retry = (first_attempt_error is not None) or (not first_attempt.answer_text.strip()) or (
            safety.call_emergency_now
            and not has_required_emergency_language(first_attempt.answer_text)
        )

        if needs_retry:
            yield self._stream_event("status", value="retrying")
            try:
                streamed_first_attempt = bool(first_attempt_state.get("streamed_tokens", False))
                if streamed_first_attempt and first_attempt.answer_text.strip():
                    retry_attempt = self._run_agent(
                        request.query,
                        model_name=request.model,
                        profile=profile,
                        top_k=request.top_k,
                        call_emergency_now=safety.call_emergency_now,
                        stricter=True,
                        history_messages=history_messages,
                    )
                else:
                    retry_state: dict[str, Any] = {}
                    async for event in self._emit_agent_attempt(
                        request.query,
                        model_name=request.model,
                        profile=profile,
                        top_k=request.top_k,
                        call_emergency_now=safety.call_emergency_now,
                        stricter=True,
                        history_messages=history_messages,
                        attempt_state=retry_state,
                    ):
                        yield event
                    retry_attempt = AgentAttemptResult(
                        answer_text=str(retry_state.get("answer_text", "")),
                        retrieval_hits=list(retry_state.get("retrieval_hits", [])),
                        used_tool=bool(retry_state.get("used_tool", False)),
                        trace_messages=list(retry_state.get("trace_messages", [])),
                    )
                overall_used_tool = retry_attempt.used_tool or overall_used_tool
                if retry_attempt.answer_text.strip():
                    final_attempt = retry_attempt
                else:
                    final_attempt = AgentAttemptResult(
                        answer_text=final_attempt.answer_text,
                        retrieval_hits=retry_attempt.retrieval_hits or final_attempt.retrieval_hits,
                        used_tool=overall_used_tool,
                        trace_messages=retry_attempt.trace_messages or final_attempt.trace_messages,
                    )
            except Exception as exc:
                warnings.append(f"Retry failed: {exc}")

        answer_text = final_attempt.answer_text
        retrieval_hits = list(final_attempt.retrieval_hits)
        trace_messages = list(final_attempt.trace_messages)

        used_fallback = False
        if not answer_text.strip():
            if not retrieval_hits:
                retrieval_hits = self.retrieve(request.query, profile, k=request.top_k)
            used_fallback = True
            answer_text = self._fallback_answer(
                request.query,
                retrieval_hits,
                safety.call_emergency_now,
            )
            warning_text = (
                "Agent returned an empty answer; used retrieval-backed fallback."
                if first_attempt_error is None
                else "Used retrieval-backed fallback after agent failure."
            )
            if warning_text not in warnings:
                warnings.append(warning_text)
            trace_messages = _compose_final_turn_trace_messages(
                request.query,
                trace_messages=[ConversationTraceMessage(role="human", content=request.query)],
                answer_text=answer_text,
            )

        if safety.call_emergency_now and not has_required_emergency_language(answer_text):
            warnings.append(
                "Emergency escalation language was injected outside the model response."
            )
            answer_text = (
                "Call emergency services now if the person is not breathing, "
                "losing consciousness, or worsening.\n\n" + answer_text
            ).strip()
        trace_messages = _compose_final_turn_trace_messages(
            request.query,
            trace_messages=trace_messages or [ConversationTraceMessage(role="human", content=request.query)],
            answer_text=answer_text,
        )

        if used_fallback and not bool(first_attempt_state.get("streamed_tokens", False)):
            yield self._stream_event("status", value="fallback")
            yield self._stream_event(
                "retrieval",
                hits=[hit.model_dump(mode="json") for hit in retrieval_hits],
            )
            yield self._stream_event("token", text=answer_text)

        for warning in warnings:
            yield self._stream_event("warning", message=warning)

        response = self._build_response(
            request,
            session_id=session_id,
            turn_id=turn_id,
            risk_category=safety.risk_category,
            call_emergency_now=safety.call_emergency_now,
            answer_text=answer_text,
            retrieval_hits=retrieval_hits,
            warnings=warnings,
            used_retrieval_tool=overall_used_tool,
        )
        self._log_response(response, trace_messages=trace_messages)

        yield self._stream_event("status", value="completed")
        yield self._stream_event("final", **response.model_dump(mode="json"))
