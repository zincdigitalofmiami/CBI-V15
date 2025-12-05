"""
AnofoxBridge - Python Bridge for TSci ↔ Anofox Integration
Phase 2.2: Creates bridge between TSci agents and Anofox SQL-native execution.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

import os

DUCKDB_PATH = Path(__file__).resolve().parents[2] / "data" / "duckdb" / "cbi_v15.duckdb"


class AnofoxBridge:
    """
    Bridge class that connects TSci agents to Anofox SQL-native execution.

    TSci makes strategic decisions (what to clean, which model, how to ensemble).
    Anofox executes heavy computations efficiently in SQL.
    """

    def __init__(
        self, duckdb_path: Optional[Path] = None, motherduck_token: str = None
    ):
        """
        Initialize AnofoxBridge with DuckDB connection.
        Prioritizes MotherDuck if token is present.

        Args:
            duckdb_path: Path to DuckDB database (defaults to standard location)
            motherduck_token: Optional MotherDuck token
        """
        self.motherduck_token = motherduck_token or os.getenv("MOTHERDUCK_TOKEN")
        self.duckdb_path = duckdb_path or DUCKDB_PATH

        if self.motherduck_token:
            logger.info("Connecting to MotherDuck (AnoFox Engine)")
            self.conn = duckdb.connect(
                f"md:cbi_v15?motherduck_token={self.motherduck_token}"
            )
        else:
            if not self.duckdb_path.parent.exists():
                self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Connecting to Local DuckDB: {self.duckdb_path}")
            self.conn = duckdb.connect(str(self.duckdb_path))

        # Try to load Anofox extensions (may not be available yet)
        try:
            self.conn.execute("LOAD anofox_tabular")
            self.conn.execute("LOAD anofox_forecast")
            self.conn.execute("LOAD anofox_statistics")
            self.anofox_available = True
        except Exception as e:
            logger.warning(f"Anofox extensions not available: {e}")
            self.anofox_available = False

    def clean_data(self, table_name: str, strategy: str = "gap_fill") -> pd.DataFrame:
        """
        Clean data using Anofox Tabular extension.
        Called by TSci Curator Agent.

        Args:
            table_name: Table name (with schema, e.g., 'raw.zl_prices')
            strategy: Cleaning strategy ('gap_fill', 'outlier_detect', etc.)

        Returns:
            Cleaned DataFrame
        """
        if not self.anofox_available:
            # Fallback: basic cleaning without Anofox
            logger.warning("Anofox not available, using basic cleaning")
            return self.conn.execute(f"SELECT * FROM {table_name}").df()

        try:
            if strategy == "gap_fill":
                # Use anofox_gap_fill
                query = f"""
                SELECT anofox_gap_fill(
                    date, close,
                    method := 'linear',
                    max_gap := '5 days'
                ) FROM {table_name}
                """
                return self.conn.execute(query).df()

            elif strategy == "outlier_detect":
                # Use anofox_outlier_detect
                query = f"""
                SELECT anofox_outlier_detect(
                    close,
                    method := 'zscore',
                    threshold := 3.0
                ) FROM {table_name}
                """
                return self.conn.execute(query).df()

            else:
                # Default: return original data
                return self.conn.execute(f"SELECT * FROM {table_name}").df()

        except Exception as e:
            logger.error(f"Error in clean_data: {e}")
            # Fallback to basic query
            return self.conn.execute(f"SELECT * FROM {table_name}").df()

    def calculate_features(self, table_name: str) -> pd.DataFrame:
        """
        Calculate features using Anofox Statistics extension.
        Called by TSci Planner Agent.

        Args:
            table_name: Table name (with schema)

        Returns:
            DataFrame with calculated features
        """
        if not self.anofox_available:
            # Fallback: basic features without Anofox
            logger.warning("Anofox not available, using basic features")
            return self.conn.execute(f"SELECT * FROM {table_name}").df()

        try:
            # Calculate features using Anofox functions
            query = f"""
            SELECT 
                date,
                close,
                anofox_volatility(close, window := 21) AS volatility_21d,
                anofox_trend_strength(close, window := 60) AS trend_60d,
                anofox_sma(close, 5) AS sma_5,
                anofox_sma(close, 20) AS sma_20,
                anofox_rsi(close, 14) AS rsi_14
            FROM {table_name}
            """
            return self.conn.execute(query).df()

        except Exception as e:
            logger.error(f"Error in calculate_features: {e}")
            # Fallback to basic query
            return self.conn.execute(f"SELECT * FROM {table_name}").df()

    def generate_baseline(
        self, table_name: str, method: str = "AutoETS", horizon: int = 30
    ) -> pd.DataFrame:
        """
        Generate baseline forecast using Anofox Forecast extension.
        Called by TSci Forecaster Agent.

        Args:
            table_name: Table name (with schema)
            method: Forecast method ('AutoETS', 'ARIMA', 'Prophet')
            horizon: Forecast horizon in periods

        Returns:
            DataFrame with forecasts
        """
        if not self.anofox_available:
            logger.warning("Anofox not available, cannot generate baseline")
            return pd.DataFrame()

        try:
            # Use TS_FORECAST function
            query = f"""
            SELECT 
                TS_FORECAST(
                    (SELECT date, close FROM {table_name}),
                    'date', 'close',
                    method := '{method}',
                    horizon := {horizon}
                ) AS forecast
            FROM {table_name}
            """
            return self.conn.execute(query).df()

        except Exception as e:
            logger.error(f"Error in generate_baseline: {e}")
            return pd.DataFrame()

    def calculate_metrics(self, actual: pd.DataFrame, predicted: pd.DataFrame) -> Dict:
        """
        Calculate forecast quality metrics using Anofox.
        Called by TSci Reporter Agent.

        Args:
            actual: DataFrame with actual values
            predicted: DataFrame with predicted values

        Returns:
            Dictionary with metrics (MAE, MSE, MAPE, etc.)
        """
        if not self.anofox_available:
            # Fallback: basic metrics calculation
            logger.warning("Anofox not available, using basic metrics")
            return self._calculate_basic_metrics(actual, predicted)

        try:
            # Use anofox_forecast_quality if available
            # For now, use basic calculation
            return self._calculate_basic_metrics(actual, predicted)

        except Exception as e:
            logger.error(f"Error in calculate_metrics: {e}")
            return self._calculate_basic_metrics(actual, predicted)

    def _calculate_basic_metrics(
        self, actual: pd.DataFrame, predicted: pd.DataFrame
    ) -> Dict:
        """Calculate basic metrics without Anofox."""
        import numpy as np

        if "value" in actual.columns and "value" in predicted.columns:
            actual_vals = actual["value"].values
            pred_vals = predicted["value"].values

            mae = np.mean(np.abs(actual_vals - pred_vals))
            mse = np.mean((actual_vals - pred_vals) ** 2)
            mape = np.mean(np.abs((actual_vals - pred_vals) / actual_vals)) * 100

            return {"mae": mae, "mse": mse, "mape": mape}

        return {"mae": 0, "mse": 0, "mape": 0}

    def close(self):
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
