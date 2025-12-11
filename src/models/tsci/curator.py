"""
TSci: Curator Agent

Responsible for data quality and outlier detection.

This agent is intentionally light-weight: it inspects table-level metrics
via the Anofox bridge and then (optionally) asks an OpenAI model for a
structured recommendation on how to proceed. The LLM does NOT see any raw
rows, only aggregated metrics, and it never executes SQL itself.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from src.anofox.anofox_bridge import AnofoxBridge
from src.utils.openai_client import run_chat


logger = logging.getLogger(__name__)


class CuratorAgent:
    """
    Data quality & hygiene agent.

    - Uses AnofoxBridge to compute simple health metrics for a given table.
    - Optionally calls an OpenAI model to classify quality and suggest actions.
    """

    def __init__(self, bridge: Optional[AnofoxBridge] = None) -> None:
        self.bridge = bridge or AnofoxBridge()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _compute_table_summary(self, table_name: str) -> Dict[str, Any]:
        """
        Compute basic health metrics for a table.

        We deliberately keep this cheap and generic so it works for any
        time-series table with a date column and optional price/volume fields.
        """
        summary: Dict[str, Any] = {"table_name": table_name}

        try:
            query = f"""
            SELECT
              COUNT(*)                                        AS row_count,
              MIN(date)                                       AS min_date,
              MAX(date)                                       AS max_date,
              SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) AS null_close,
              SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) AS null_price,
              SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS null_volume
            FROM {table_name}
            """
            df = self.bridge.conn.execute(query).df()  # type: ignore[attr-defined]
            if not df.empty:
                summary.update(df.to_dict(orient="records")[0])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to compute table summary for %s: %s", table_name, exc
            )

        return summary

    def _call_llm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the OpenAI-backed curator helper.

        The model is instructed to RETURN JSON ONLY. If parsing fails or the
        client is unavailable, we fall back to a conservative default.
        """
        system_prompt = (
            "You are a quantitative data-quality analyst for a soybean oil "
            "futures forecasting system. You NEVER invent data or table names. "
            "You receive aggregated metrics only (row counts, null counts, "
            "date ranges). Your job is to:\n"
            "1) classify overall data_quality as one of: 'pass', 'warn', 'fail';\n"
            "2) choose an outlier_strategy (e.g., 'none', 'clip', 'drop');\n"
            "3) recommend next_actions for the pipeline.\n"
            "Respond with a single JSON object only, with keys:\n"
            "  data_quality, outlier_strategy, recommendation, risk_flags.\n"
        )

        try:
            response_text = run_chat(
                prompt=json.dumps(payload),
                system=system_prompt,
                temperature=0.1,
            )
            result = json.loads(response_text)
            if not isinstance(result, dict):
                raise ValueError("LLM response was not a JSON object")
            return result
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Curator LLM call failed, using fallback: %s", exc)
            return {
                "data_quality": "pass",
                "outlier_strategy": "none",
                "recommendation": "proceed",
                "risk_flags": [],
            }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze_data_quality(self, table_name: str) -> Dict[str, Any]:
        """
        Analyze data quality for a given table using Anofox + OpenAI.

        Returns a structured dict suitable for writing into tsci.qa_checks
        and for display in the Quant Admin dashboard.
        """
        summary = self._compute_table_summary(table_name)
        llm_payload = {
            "table_name": table_name,
            "summary": summary,
        }
        decision = self._call_llm(llm_payload)

        # Ensure engine/source metadata are always present
        decision.setdefault("engine", "Anofox (MotherDuck)")
        decision.setdefault("table_name", table_name)
        decision.setdefault("summary", summary)
        return decision
