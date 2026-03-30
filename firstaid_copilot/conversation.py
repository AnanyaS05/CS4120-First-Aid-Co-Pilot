from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .schemas import (
    ConversationMessage,
    ConversationSummary,
    ConversationThread,
    ConversationTraceMessage,
    ConversationTurnRecord,
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def make_session_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"session-{timestamp}-{uuid4().hex[:8]}"


def make_turn_id() -> str:
    return f"turn-{uuid4().hex[:10]}"


def sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned or f"session-{uuid4().hex[:8]}"


def _truncate_text(text: str, *, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


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


def _strip_tool_markup(text: str) -> str:
    cleaned = re.sub(
        r"<search_first_aid_knowledge>.*?</search_first_aid_knowledge>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"</?search_first_aid_knowledge>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def trace_message_from_langchain(message: BaseMessage) -> ConversationTraceMessage | None:
    if isinstance(message, HumanMessage):
        content = _content_to_text(message.content)
        if not content:
            return None
        return ConversationTraceMessage(
            role="human",
            content=content,
            name=message.name,
        )

    if isinstance(message, ToolMessage):
        return ConversationTraceMessage(
            role="tool",
            content=_content_to_text(message.content),
            name=message.name,
            tool_call_id=message.tool_call_id,
        )

    if isinstance(message, AIMessage):
        content = _strip_tool_markup(_content_to_text(message.content))
        tool_calls = list(message.tool_calls or [])
        if not content and not tool_calls:
            return None
        return ConversationTraceMessage(
            role="assistant",
            content=content,
            name=message.name,
            tool_calls=tool_calls or None,
        )

    return None


def langchain_message_from_trace(message: ConversationTraceMessage) -> BaseMessage:
    if message.role == "human":
        return HumanMessage(content=message.content, name=message.name)
    if message.role == "assistant":
        return AIMessage(
            content=message.content,
            name=message.name,
            tool_calls=list(message.tool_calls or []),
        )
    return ToolMessage(
        content=message.content,
        name=message.name,
        tool_call_id=message.tool_call_id or "persisted-tool-call",
    )


@dataclass(slots=True)
class ConversationLogger:
    conversations_dir: Path

    def __post_init__(self) -> None:
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        safe_session_id = sanitize_identifier(session_id)
        return self.conversations_dir / f"{safe_session_id}.jsonl"

    def _session_files(self) -> list[Path]:
        if not self.conversations_dir.exists():
            return []
        return sorted(
            [
                path
                for path in self.conversations_dir.glob("*.jsonl")
                if path.name != "runs.jsonl"
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def session_exists(self, session_id: str) -> bool:
        return self._session_path(session_id).exists()

    def _load_session_records(self, session_id: str) -> list[ConversationTurnRecord]:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Conversation '{session_id}' was not found.")

        records: list[ConversationTurnRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(ConversationTurnRecord.model_validate(json.loads(line)))
        return records

    def log_turn(self, session_id: str, payload: dict) -> Path:
        record = ConversationTurnRecord.model_validate(payload)
        path = self._session_path(session_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")
        return path

    def log_run(self, payload: dict) -> Path:
        path = self.conversations_dir / "runs.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
        return path

    def list_conversations(self) -> list[ConversationSummary]:
        summaries: list[ConversationSummary] = []
        for path in self._session_files():
            session_id = path.stem
            records = self._load_session_records(session_id)
            if not records:
                continue

            first = records[0]
            last = records[-1]
            title = _truncate_text(first.user_query or session_id, limit=56)
            preview_source = last.final_answer or last.user_query
            preview = _truncate_text(preview_source, limit=84)
            summaries.append(
                ConversationSummary(
                    session_id=session_id,
                    title=title or session_id,
                    preview=preview,
                    updated_at=last.timestamp,
                    turn_count=len(records),
                    model=last.model,
                    profile=last.profile,
                )
            )

        return sorted(summaries, key=lambda item: item.updated_at, reverse=True)

    def load_context_messages(
        self,
        session_id: str,
        *,
        max_human_messages: int,
    ) -> list[BaseMessage]:
        if not self.session_exists(session_id):
            return []

        records = self._load_session_records(session_id)
        trace_messages = [
            trace_message
            for record in records
            for trace_message in record.trace_messages
        ]
        human_indices = [
            index
            for index, trace_message in enumerate(trace_messages)
            if trace_message.role == "human"
        ]
        if len(human_indices) > max_human_messages:
            trace_messages = trace_messages[human_indices[-max_human_messages] :]

        return [
            langchain_message_from_trace(trace_message)
            for trace_message in trace_messages
        ]

    def load_conversation(self, session_id: str) -> ConversationThread:
        records = self._load_session_records(session_id)
        if not records:
            raise FileNotFoundError(f"Conversation '{session_id}' was not found.")

        messages: list[ConversationMessage] = []
        for index, record in enumerate(records, start=1):
            timestamp = record.timestamp
            turn_id = record.turn_id or f"turn-{index}"
            user_query = record.user_query.strip()
            assistant_text = record.final_answer.strip()

            if user_query:
                messages.append(
                    ConversationMessage(
                        id=f"{turn_id}-user",
                        role="user",
                        text=user_query,
                        timestamp=timestamp,
                    )
                )
            if assistant_text:
                messages.append(
                    ConversationMessage(
                        id=f"{turn_id}-assistant",
                        role="assistant",
                        text=assistant_text,
                        timestamp=timestamp,
                        sources=list(record.retrieval_hits) or None,
                        warnings=list(record.warnings) or None,
                    )
                )

        title = _truncate_text(records[0].user_query or session_id, limit=56)
        return ConversationThread(
            session_id=sanitize_identifier(session_id),
            title=title or sanitize_identifier(session_id),
            messages=messages,
        )
