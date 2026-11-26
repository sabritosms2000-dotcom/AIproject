"""
Person 3 Integration Interfaces (AI Assistant & Analytics Dashboard).

Provides clean adapter protocols and honest non-fabricated fallbacks so the
application runs standalone until Person 3 connects their AI Agent and Analytics.
"""

from __future__ import annotations

from typing import Any, Protocol
import pandas as pd

from signbridge.config import RESULTS_CSV
from signbridge.history import TranslationHistoryStore


class AssistantAdapter(Protocol):
    """Protocol for Person 3's AI Assistant integration."""

    def is_connected(self) -> bool: ...
    def status_message(self) -> str: ...
    def ask(self, user_query: str, context: dict[str, Any] | None = None) -> str: ...


class AnalyticsAdapter(Protocol):
    """Protocol for Person 3's Analytics Dashboard integration."""

    def is_connected(self) -> bool: ...
    def get_dashboard_data(self) -> dict[str, Any]: ...


class PlaceholderAssistantAdapter:
    """Honest fallback when Person 3's AI Agent is not yet connected."""

    def is_connected(self) -> bool:
        return False

    def status_message(self) -> str:
        return (
            "Person 3 AI Assistant is not connected yet. "
            "Implement `AssistantAdapter.ask(user_query, context)` to integrate."
        )

    def ask(self, user_query: str, context: dict[str, Any] | None = None) -> str:
        ctx_keys = ", ".join(context.keys()) if context else "none"
        return (
            "[Person 3 Integration Placeholder]\n"
            f"Received query: {user_query}\n"
            f"Available context keys: {ctx_keys}\n"
            "No AI Agent backend is attached in Person 2's deliverable."
        )


class LocalAnalyticsAdapter:
    """Reads only real verified model metrics (lstm_v2_results.csv) and local SQLite history."""

    def __init__(self, history_store: TranslationHistoryStore) -> None:
        self.history_store = history_store

    def is_connected(self) -> bool:
        return False  # Person 3 custom analytics module not attached

    def get_dashboard_data(self) -> dict[str, Any]:
        model_rows = []
        if RESULTS_CSV.exists():
            df = pd.read_csv(RESULTS_CSV)
            model_rows = df.to_dict(orient="records")

        return {
            "person3_connected": False,
            "model_results": model_rows,
            "history_stats": self.history_store.get_summary_stats(),
        }
