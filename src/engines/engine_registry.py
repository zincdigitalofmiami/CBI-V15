"""
Engine Registry - Multi-Engine Support Architecture
Phase 2.5: Registry for managing multiple forecasting engines.

Supports model registration for orchestration layers to use:
- AutoGluon-based engines (primary, V15.1)
- Legacy baselines (LightGBM/CatBoost/XGBoost) for comparison
"""

from typing import Dict, List, Optional

import pandas as pd

from .base_engine import BaseEngine
import logging

logger = logging.getLogger(__name__)

# Model family registry (name -> import path)
# Note: These are **legacy baselines** only; the canonical V15.1 stack uses AutoGluon.
MODEL_FAMILIES = {
    "lightgbm": "src.training.baselines.lightgbm_zl",
    "catboost": "src.training.baselines.catboost_zl",
    "xgboost": "src.training.baselines.xgboost_zl",
    # Future legacy models (if ever added, for experiments only):
    # "tft": "src.training.deep.tft_zl",
    # "prophet": "src.training.statistical.prophet_zl",
    # "garch": "src.training.statistical.garch_zl",
}


class EngineRegistry:
    """
    Registry for managing multiple forecasting engines.

    Supports:
    - Engine registration
    - Engine selection based on regime/horizon
    - Ensemble combination of multiple engines
    """

    def __init__(self):
        """Initialize engine registry."""
        self.engines: Dict[str, BaseEngine] = {}
        self.engine_weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}

    def register(self, engine: BaseEngine, default_weight: float = 1.0):
        """
        Register an engine.

        Args:
            engine: Engine instance (must implement BaseEngine)
            default_weight: Default weight for ensemble (1.0 = equal weight)
        """
        self.engines[engine.name] = engine
        self.engine_weights[engine.name] = default_weight
        self.performance_history[engine.name] = []
        logger.info(f"Registered engine: {engine.name} (weight: {default_weight})")

    def get_engine(self, name: str) -> Optional[BaseEngine]:
        """
        Get engine by name.

        Args:
            name: Engine name

        Returns:
            Engine instance or None if not found
        """
        return self.engines.get(name)

    @staticmethod
    def get_model_trainer(model_family: str):
        """
        Get the training module for a model family by name.

        Args:
            model_family: Model family name (lightgbm, catboost, xgboost)

        Returns:
            Training module or None if not found

        Example:
            >>> trainer = EngineRegistry.get_model_trainer("catboost")
            >>> models, metrics = trainer.train_catboost_for_horizon(...)
        """
        if model_family not in MODEL_FAMILIES:
            logger.warning(f"Model family '{model_family}' not registered")
            return None

        try:
            import importlib

            module_path = MODEL_FAMILIES[model_family]
            module = importlib.import_module(module_path)
            return module
        except ImportError as exc:
            logger.error(f"Failed to import {model_family}: {exc}")
            return None

    def list_engines(self) -> List[str]:
        """List all registered engine names."""
        return list(self.engines.keys())

    def select_engines(self, regime: str, horizon: int) -> List[str]:
        """
        Select appropriate engines based on regime and horizon.

        Args:
            regime: Market regime (e.g., 'high_volatility', 'trending')
            horizon: Forecast horizon in periods

        Returns:
            List of engine names to use
        """
        # Default: use all engines
        # Subclasses can override for regime-specific selection
        selected = []

        # High volatility regimes: prefer statistical / robust models
        if regime in ["high_volatility", "crisis"]:
            if "anofox" in self.engines:
                selected.append("anofox")

        # Trending regimes: prefer ML models
        elif regime in ["trending", "bull", "bear"]:
            if "chronos2" in self.engines:
                selected.append("chronos2")
            if "autogluon" in self.engines:
                selected.append("autogluon")

        # Short horizons: prefer fast models
        if horizon <= 7:
            if "anofox" in self.engines:
                selected.append("anofox")

        # Long horizons: prefer robust models
        elif horizon >= 90:
            if "autogluon" in self.engines:
                selected.append("autogluon")

        # Default: use all available engines
        if not selected:
            selected = list(self.engines.keys())

        return selected

    def update_performance(self, engine_name: str, mape: float):
        """
        Update performance history for an engine.

        Args:
            engine_name: Engine name
            mape: Mean Absolute Percentage Error
        """
        if engine_name in self.performance_history:
            self.performance_history[engine_name].append(mape)
            # Keep only last 100 performance records
            if len(self.performance_history[engine_name]) > 100:
                self.performance_history[engine_name] = self.performance_history[
                    engine_name
                ][-100:]

    def get_weights(self, engine_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get ensemble weights for engines.

        Args:
            engine_names: Optional list of engine names (None = all engines)

        Returns:
            Dictionary mapping engine names to weights
        """
        if engine_names is None:
            engine_names = list(self.engines.keys())

        # Calculate weights based on recent performance
        weights = {}
        total_weight = 0.0

        for name in engine_names:
            if name not in self.engines:
                continue

            # Base weight
            base_weight = self.engine_weights.get(name, 1.0)

            # Adjust based on recent performance (lower MAPE = higher weight)
            if name in self.performance_history and self.performance_history[name]:
                recent_mape = sum(self.performance_history[name][-10:]) / len(
                    self.performance_history[name][-10:]
                )
                # Invert MAPE (lower is better) and normalize
                performance_weight = 1.0 / (1.0 + recent_mape)
            else:
                performance_weight = 1.0

            weight = base_weight * performance_weight
            weights[name] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            for name in weights:
                weights[name] /= total_weight

        return weights

    def combine_forecasts(
        self,
        forecasts: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Combine forecasts from multiple engines using weighted average.

        Args:
            forecasts: Dictionary mapping engine names to forecast DataFrames
            weights: Optional weights (None = use registry weights)

        Returns:
            Combined forecast DataFrame
        """
        if not forecasts:
            raise ValueError("No forecasts provided")

        if weights is None:
            weights = self.get_weights(list(forecasts.keys()))

        # Get first forecast to use as template
        first_engine = list(forecasts.keys())[0]
        combined = forecasts[first_engine].copy()

        # Initialize weighted sum
        if "forecast" in combined.columns:
            combined["forecast"] = combined["forecast"] * weights.get(first_engine, 0)

        # Add weighted contributions from other engines
        for engine_name, forecast_df in forecasts.items():
            if engine_name == first_engine:
                continue

            weight = weights.get(engine_name, 0)
            if weight > 0 and "forecast" in forecast_df.columns:
                # Align dates and add weighted forecast
                merged = combined.merge(
                    forecast_df[["date", "forecast"]],
                    on="date",
                    how="outer",
                    suffixes=("", f"_{engine_name}"),
                )
                combined["forecast"] = (
                    combined["forecast"]
                    + merged[f"forecast_{engine_name}"].fillna(0) * weight
                )

        return combined
