"""
TSci: Reporter Agent

Responsible for generating narrative reports for the Quant Admin dashboard.

Uses OpenAI to generate web-ready HTML reports and structured JSON from
forecast distributions, bucket contributions, and risk metrics. The LLM
creates narrative explanations; all numbers come from MotherDuck.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.utils.openai_client import run_chat

logger = logging.getLogger(__name__)


class ReporterAgent:
    """
    Performance reporting and narrative generation agent.

    - Retrieves forecast results and diagnostics from MotherDuck.
    - Uses OpenAI to generate structured, web-ready reports.
    - Outputs HTML + JSON suitable for /quant-admin dashboard.
    """

    def __init__(self) -> None:
        pass

    def generate_report(
        self,
        run_id: str,
        forecast_data: Optional[Dict[str, Any]] = None,
        bucket_contributions: Optional[Dict[str, float]] = None,
        risk_metrics: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a web-ready report for a given run.

        Args:
            run_id: Unique run identifier
            forecast_data: Forecast distributions (P10/P50/P90, etc.)
            bucket_contributions: Big 8 bucket impact scores
            risk_metrics: Downside risk, scenarios from Monte Carlo
            use_llm: Whether to use OpenAI for narrative generation

        Returns:
            Dict with run_id, summary_html, drivers, scenarios, confidence
        """
        if not use_llm:
            # Fallback: simple template
            return {
                "run_id": run_id,
                "summary_html": "<p>Automated run completed successfully.</p>",
                "drivers": [],
                "scenarios": [],
                "confidence": "medium",
                "next_actions": "monitor",
            }

        system_prompt = (
            "You are a risk-aware quant analyst explaining ZL futures forecasts to a "
            "commodity risk manager. You receive forecast distributions, bucket contributions, "
            "and risk metrics. Generate a structured report with:\n"
            "1) summary_html: 1-2 paragraph HTML snippet (no <html> tags, just <p>)\n"
            "2) drivers: array of {bucket, impact, explanation} for top drivers\n"
            "3) scenarios: array of {name, description, probability} for key risks\n"
            "4) confidence: 'high' | 'medium' | 'low'\n"
            "5) next_actions: brief action items\n"
            "RETURN JSON ONLY. NEVER invent numbers not in the input. "
            "The Big 8 buckets are: Crush, China, FX, Fed, Tariff, Biofuel, Energy, Volatility "
            "(note: Volatility = VIX/vol regimes, NOT trading volume)."
        )

        payload = {
            "run_id": run_id,
            "forecast_data": forecast_data or {},
            "bucket_contributions": bucket_contributions or {},
            "risk_metrics": risk_metrics or {},
        }

        try:
            response_text = run_chat(
                prompt=json.dumps(payload),
                system=system_prompt,
                temperature=0.3,
            )
            result = json.loads(response_text)
            if not isinstance(result, dict):
                raise ValueError("LLM response was not JSON")
            # Ensure run_id is always present
            result["run_id"] = run_id
            return result
        except Exception as exc:
            logger.warning("Reporter LLM call failed, using fallback: %s", exc)
            return {
                "run_id": run_id,
                "summary_html": "<p>Forecast run completed. Review metrics in Quant Admin.</p>",
                "drivers": [],
                "scenarios": [],
                "confidence": "medium",
                "next_actions": "monitor",
            }
