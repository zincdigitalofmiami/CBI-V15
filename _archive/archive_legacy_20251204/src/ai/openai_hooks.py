"""
OpenAI Hooks for TSci Agents - AI-Powered Enhancements
Phase 2.7: Shared OpenAI utilities for TSci agent AI capabilities.
"""

import os
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. Install with: pip install openai")


class OpenAIHooks:
    """
    Shared OpenAI client and utilities for TSci agent AI hooks.
    
    Provides:
    - Redundancy detection for Curator Agent
    - Experiment design for Planner Agent
    - Narrative generation for Reporter Agent
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize OpenAI hooks.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def detect_redundancy(self, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect redundant data patterns using AI.
        Called by TSci Curator Agent.
        
        Args:
            data_summary: Dictionary with data statistics (row counts, null rates, etc.)
        
        Returns:
            Dictionary with redundancy analysis and recommendations
        """
        prompt = f"""
        Analyze the following data quality summary and identify:
        1. Duplicate or redundant data patterns
        2. Redundant features that provide similar information
        3. Data cleaning strategies to remove redundancy
        
        Data Summary:
        {data_summary}
        
        Provide a structured analysis with:
        - Redundant patterns identified
        - Feature redundancy scores
        - Recommended cleaning strategies
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data quality expert specializing in identifying redundant patterns in time-series financial data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            
            return {
                'redundancy_detected': True,
                'analysis': analysis,
                'recommendations': self._parse_recommendations(analysis)
            }
        
        except Exception as e:
            logger.error(f"Error in detect_redundancy: {e}")
            return {
                'redundancy_detected': False,
                'error': str(e)
            }
    
    def design_experiment(self, 
                          current_features: List[str],
                          target: str,
                          regime: str,
                          horizon: int) -> Dict[str, Any]:
        """
        Design forecasting experiment using AI.
        Called by TSci Planner Agent.
        
        Args:
            current_features: List of current feature names
            target: Target variable name
            regime: Current market regime
            horizon: Forecast horizon
        
        Returns:
            Dictionary with experiment design (feature combinations, model architectures, etc.)
        """
        prompt = f"""
        Design a forecasting experiment for:
        - Target: {target}
        - Regime: {regime}
        - Horizon: {horizon} periods
        
        Current Features ({len(current_features)}):
        {', '.join(current_features[:20])}...
        
        Provide:
        1. Recommended feature combinations
        2. Model architectures to test
        3. Hyperparameter search spaces
        4. Experiment hypotheses
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a quantitative finance expert specializing in time-series forecasting experiments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            
            design = response.choices[0].message.content
            
            return {
                'experiment_designed': True,
                'design': design,
                'recommendations': self._parse_experiment_design(design)
            }
        
        except Exception as e:
            logger.error(f"Error in design_experiment: {e}")
            return {
                'experiment_designed': False,
                'error': str(e)
            }
    
    def generate_narrative(self,
                          forecast_results: Dict[str, Any],
                          model_performance: Dict[str, float]) -> str:
        """
        Generate human-readable forecast narrative using AI.
        Called by TSci Reporter Agent.
        
        Args:
            forecast_results: Dictionary with forecast results
            model_performance: Dictionary with model metrics (MAE, MAPE, etc.)
        
        Returns:
            Narrative text explaining forecast results
        """
        prompt = f"""
        Generate a clear, concise narrative explaining these forecast results:
        
        Forecast Results:
        {forecast_results}
        
        Model Performance:
        {model_performance}
        
        Include:
        1. Executive summary
        2. Key insights
        3. Model decisions explained
        4. Confidence assessment
        5. Next actions
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst writing clear, professional reports for commodity traders."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error in generate_narrative: {e}")
            return f"Error generating narrative: {e}"
    
    def _parse_recommendations(self, analysis: str) -> List[str]:
        """Parse recommendations from AI analysis."""
        # Simple parsing - can be enhanced
        lines = analysis.split('\n')
        recommendations = []
        for line in lines:
            if 'recommend' in line.lower() or 'suggest' in line.lower() or line.strip().startswith('-'):
                recommendations.append(line.strip())
        return recommendations
    
    def _parse_experiment_design(self, design: str) -> Dict[str, Any]:
        """Parse experiment design from AI response."""
        # Simple parsing - can be enhanced
        return {
            'raw_design': design,
            'features': [],
            'models': [],
            'hyperparameters': {}
        }

