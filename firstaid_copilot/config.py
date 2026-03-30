from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

ProfileName = Literal["experiment", "demo"]

SUPPORTED_MODELS = (
    "functiongemma",
    "qwen3:0.6b",
    "qwen3.5:0.8b",
    "granite4:350m",
)
SUPPORTED_PROFILES = ("experiment", "demo")


def normalize_model_name(model_name: str) -> str:
    model_name = model_name.strip().casefold()
    if model_name.endswith(":latest"):
        return model_name[: -len(":latest")]
    return model_name


@dataclass(slots=True)
class AppConfig:
    root_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("FIRSTAID_DEFAULT_MODEL", "qwen3:0.6b")
    )
    default_agent_top_k: int = 5
    default_debug_top_k: int = 5
    request_timeout_seconds: float = 60.0
    model_names: tuple[str, ...] = SUPPORTED_MODELS

    def __post_init__(self) -> None:
        self.root_dir = self.root_dir.resolve()

    @property
    def preprocessing_dir(self) -> Path:
        return self.root_dir / "preprocessing"

    @property
    def artifacts_dir(self) -> Path:
        return self.root_dir / "artifacts"

    @property
    def indexes_dir(self) -> Path:
        return self.artifacts_dir / "indexes"

    @property
    def conversations_dir(self) -> Path:
        return self.artifacts_dir / "conversations"

    def index_dir(self, profile: ProfileName) -> Path:
        return self.indexes_dir / profile

    def ensure_runtime_dirs(self) -> None:
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def validate_profile(self, profile: str) -> ProfileName:
        if profile not in SUPPORTED_PROFILES:
            raise ValueError(
                f"Unsupported profile '{profile}'. Expected one of {SUPPORTED_PROFILES}."
            )
        return profile  # type: ignore[return-value]

    def validate_model(self, model_name: str) -> str:
        normalized = normalize_model_name(model_name)
        supported = {normalize_model_name(name) for name in self.model_names}
        if normalized not in supported:
            raise ValueError(
                f"Unsupported model '{model_name}'. Expected one of {self.model_names}."
            )
        return model_name
