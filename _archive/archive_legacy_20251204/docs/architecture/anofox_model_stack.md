# AnoFox Model Stack

**Status:** Production  
**Last Updated:** December 3, 2025

## AutoML Strategy

31 forecasting models trained on ZL data across 5 horizons (1W, 1M, 3M, 6M, 12M).

## Model Categories

**Baseline (4):** Naive, SeasonalNaive, WindowAverage, RandomWalkWithDrift

**Exponential Smoothing (9):** SimpleExponentialSmoothing, Holt, HoltWinters, ETS, AutoETS, Croston, IMAPA, TSB, ADIDA

**ARIMA (3):** ARIMA, AutoARIMA, ARIMA+GARCH

**Theta (4):** Theta, OptimizedTheta, DynamicTheta, DynamicOptimizedTheta

**Advanced Seasonal (6):** TBATS, BATS, MSTL, STL, STLDecomposition, MFLES

**Special (5):** HistoricAverage, SeasonalWindowAverage, CrostonSBA, CrostonOptimized, AutoCES

## Ensemble

Top 3-5 models per horizon selected based on MAPE.

**Weighting:** Inverse MAPE, recalculated weekly.

## Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Bias
- Coverage (90%, 95% prediction intervals)

