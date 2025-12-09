# Features

## Purpose
Feature engineering code - Python implementations of technical indicators and feature calculations.

## What Belongs Here
- `technical_indicators_incremental.py` - Streaming technical indicators
- Custom feature calculators
- Feature validation utilities

## What Does NOT Belong Here
- Feature SQL definitions (→ `database/definitions/03_features/`)
- Feature documentation (→ `docs/features/`)

## Relationship to Database
- SQL in `database/definitions/03_features/` defines how features are STORED
- Python here defines how features are COMPUTED (for streaming/incremental)
- For batch processing, prefer SQL; for streaming, use Python

## Current Files
- `technical_indicators_incremental.py` - RSI, MACD, Bollinger Bands, etc. for streaming data

