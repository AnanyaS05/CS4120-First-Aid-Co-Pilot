from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


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


@dataclass(slots=True)
class ConversationLogger:
    conversations_dir: Path

    def __post_init__(self) -> None:
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def log_turn(self, session_id: str, payload: dict) -> Path:
        safe_session_id = sanitize_identifier(session_id)
        path = self.conversations_dir / f"{safe_session_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
        return path

    def log_run(self, payload: dict) -> Path:
        path = self.conversations_dir / "runs.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
        return path
