"""
TSci: Forecaster Agent

Responsible for model selection, ensemble weighting, and forecasting.

Uses OpenAI to guide ensemble strategy (QRA weighting) based on regime and
validation metrics. The LLM suggests weights; numeric execution happens in
our own QRA and Monte Carlo modules.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.engines.anofox.anofox_bridge import AnofoxBridge
from src.ensemble.qra_ensemble import EnsembleForecast, QuantileForecast, run_qra
from src.models.tsci.model_sweep import SweepConfig, run_model_sweep, select_best_model
from src.simulators.monte_carlo_sim import SimulationConfig, simulate_paths
from src.utils.openai_client import run_chat

logger = logging.getLogger(__name__)


class ForecasterAgent:
    """
    Model selection and ensemble orchestration agent.

    - Gathers per-model quantile forecasts from L1/L2.
    - Uses OpenAI to suggest QRA ensemble weights based on regime.
    - Executes QRA and Monte Carlo numerically (not via LLM).
    """

    def __init__(self, bridge: Optional[AnofoxBridge] = None) -> None:
        self.bridge = bridge or AnofoxBridge()

    def run_bucket_sweep(
        self,
        bucket: str,
        horizon: str,
        candidate_models: List[str],
        train_table: str = "training.daily_ml_matrix_zl",
        regime: str = "adaptive",
    ) -> Dict[str, Any]:
        """
        Run a model sweep for a specific bucket and horizon.

        Args:
            bucket: Bucket name (e.g., 'volatility', 'crush')
            horizon: Horizon (e.g., '1w', '1m')
            candidate_models: Model families to try
            train_table: Training table in MotherDuck
            regime: Market regime

        Returns:
            Dict with sweep_results, winner, metrics
        """
        config = SweepConfig(
            horizon=horizon,
            bucket_focus=bucket,
            candidate_models=candidate_models,
            train_table=train_table,
            regime=regime,
        )

        # Run sweep
        results = run_model_sweep(config)

        # Select best model
        winner = select_best_model(results, metric="pinball_p50")

        return {
            "sweep_config": config.to_dict(),
            "results": [r.to_dict() for r in results],
            "winner": winner.to_dict() if winner else None,
            "regime": regime,
        }

    def suggest_ensemble_weights(
        self,
        models: List[str],
        regime: str,
        metrics: Dict[str, Dict[str, float]],
        use_llm: bool = True,
    ) -> Dict[str, float]:
        """
        Suggest ensemble weights for QRA.

        Args:
            models: List of model names
            regime: Current market regime
            metrics: Dict mapping model names to their validation metrics
            use_llm: Whether to use OpenAI for suggestions

        Returns:
            Dict mapping model names to weights (sum to 1.0)
        """
        if not use_llm or not models:
            # Fallback: equal weights
            return {model: 1.0 / len(models) for model in models}

        system_prompt = (
            "You are a quantitative analyst specializing in commodity macro forecasting. "
            "Given model validation metrics and current market regime, recommend optimal "
            "weights for a Quantile Regression Averaging (QRA) ensemble. "
            "Preserve the uncertainty structure from quantiles. "
            "RETURN JSON ONLY with key 'weights' mapping model names to floats (must sum to ~1.0). "
            "NEVER invent models or metrics not in the input."
        )

        payload = {
            "models": models,
            "regime": regime,
            "metrics": metrics,
        }

        try:
            response_text = run_chat(
                prompt=json.dumps(payload),
                system=system_prompt,
                temperature=0.1,
            )
            result = json.loads(response_text)
            if "weights" in result and isinstance(result["weights"], dict):
                # Normalize to sum to 1.0
                total = sum(result["weights"].values())
                if total > 0:
                    return {k: v / total for k, v in result["weights"].items()}
            raise ValueError("LLM response missing valid 'weights'")
        except Exception as exc:
            logger.warning("Forecaster LLM call failed, using equal weights: %s", exc)
            return {model: 1.0 / len(models) for model in models}

    def generate_forecast(
        self,
        features_table: str,
        horizon: str,
        candidate_models: Optional[List[str]] = None,
        regime: str = "adaptive",
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate forecast for a given horizon using trained models + QRA + Monte Carlo.

        Args:
            features_table: Table with features
            horizon: Forecast horizon (1w, 1m, 3m, 6m, 12m)
            candidate_models: Model names to ensemble
            regime: Current regime
            use_llm: Whether to use OpenAI for ensemble weighting

        Returns:
            Dict with forecast_table, ensemble_strategy, weights, etc.
        """
        if candidate_models is None:
            candidate_models = ["lightgbm", "catboost"]

        # TODO: In production, fetch actual validation metrics from tsci.runs
        mock_metrics = {
            model: {"rmse": 0.5, "pinball": 0.3, "coverage": 0.9}
            for model in candidate_models
        }

        # Get ensemble weights (LLM-assisted)
        weights = self.suggest_ensemble_weights(
            models=candidate_models,
            regime=regime,
            metrics=mock_metrics,
            use_llm=use_llm,
        )

        # TODO: Execute QRA numerically via src/ensemble/qra_ensemble.py
        # TODO: Execute Monte Carlo via src/simulators/monte_carlo_sim.py

        return {
            "primary_models": candidate_models,
            "ensemble_strategy": "qra",
            "ensemble_weights": weights,
            "forecast_table": "forecasts.zl_predictions",
            "engine": "Anofox (MotherDuck)",
            "regime": regime,
        }

    def run_full_forecast_pipeline(
        self,
        quantile_forecasts: List[QuantileForecast],
        weights: Dict[str, float],
        regime: str = "adaptive",
        run_monte_carlo: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full L3+L4 pipeline: QRA ensemble → Monte Carlo simulation.

        Args:
            quantile_forecasts: List of QuantileForecast from L1/L2 models
            weights: Model weights for QRA
            regime: Current market regime
            run_monte_carlo: Whether to run Monte Carlo (L4)

        Returns:
            Dict with ensemble_forecast, simulation_result, risk_metrics
        """
        logger.info("\n🔄 Running Full Forecast Pipeline (L3 → L4)")

        # L3: QRA Ensemble
        ensemble = run_qra(
            forecasts=quantile_forecasts,
            weights=weights,
            regime=regime,
        )

        result = {
            "ensemble_forecast": ensemble.to_dataframe(),
            "ensemble_weights": weights,
            "regime": regime,
        }

        # L4: Monte Carlo (optional)
        if run_monte_carlo:
            sim_result = simulate_paths(
                p10=ensemble.p10,
                p50=ensemble.p50,
                p90=ensemble.p90,
                dates=ensemble.dates,
                horizon=ensemble.horizon,
                config=SimulationConfig(n_paths=1000),
            )

            result["simulation"] = sim_result.to_dict()
            result["risk_metrics"] = sim_result.downside_risk
            result["scenario_stats"] = sim_result.scenario_stats

        return result
