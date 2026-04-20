"""First-Aid Co-Pilot package."""

# Re-export the main config and service for convenient imports.

from .config import AppConfig
from .service import FirstAidCopilotService

__all__ = ["AppConfig", "FirstAidCopilotService"]

