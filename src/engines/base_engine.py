"""
Base Engine Interface - Multi-Engine Support Architecture
Phase 2.5: Abstract base class for all forecasting engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import pandas as pd
from pathlib import Path


class BaseEngine(ABC):
    """
    Base class for all forecasting engines.

    All engines (Anofox, AutoGluon, and any legacy baselines) must implement this interface.
    TSci is no longer the primary orchestrator; it is optional/legacy.
    """
    
    def __init__(self, name: str, model_dir: Optional[Path] = None):
        """
        Initialize engine.

        Args:
            name: Engine name (e.g., 'anofox', 'autogluon', 'chronos2', 'baseline_lightgbm')
            model_dir: Directory to store model artifacts
        """
        self.name = name
        self.model_dir = model_dir or Path(f"models/{name}")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def train(self, 
              train_data: pd.DataFrame,
              target_column: str,
              feature_columns: Optional[list] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training DataFrame
            target_column: Name of target column
            feature_columns: Optional list of feature columns (None = all except target)
            **kwargs: Engine-specific hyperparameters
        
        Returns:
            Dictionary with training metrics and model info
        """
        pass
    
    @abstractmethod
    def predict(self,
                data: pd.DataFrame,
                horizon: int = 30,
                **kwargs) -> pd.DataFrame:
        """
        Generate predictions.
        
        Args:
            data: Input DataFrame with features
            horizon: Forecast horizon (number of periods)
            **kwargs: Engine-specific prediction parameters
        
        Returns:
            DataFrame with predictions (columns: date, forecast, lower_bound, upper_bound)
        """
        pass
    
    @abstractmethod
    def evaluate(self,
                 actual: pd.DataFrame,
                 predicted: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            actual: DataFrame with actual values
            predicted: DataFrame with predicted values
        
        Returns:
            Dictionary with metrics (MAE, MSE, MAPE, R², etc.)
        """
        pass
    
    def save_model(self, model_id: str) -> Path:
        """
        Save model artifacts.
        
        Args:
            model_id: Unique model identifier
        
        Returns:
            Path to saved model
        """
        model_path = self.model_dir / f"{model_id}.pkl"
        # Subclasses should implement actual saving logic
        return model_path
    
    def load_model(self, model_path: Path):
        """
        Load model artifacts.
        
        Args:
            model_path: Path to saved model
        """
        # Subclasses should implement actual loading logic
        pass
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with columns: feature, importance
        """
        # Default: return empty DataFrame
        # Subclasses should override if they support feature importance
        return pd.DataFrame(columns=['feature', 'importance'])
