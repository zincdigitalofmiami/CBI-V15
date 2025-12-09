# TSci: The Intelligence Framework ("The Brain")

TSci is the **System 2 Thinking** layer of the platform. It is a collection of OpenAI-powered "Agents" that make strategic decisions, while delegating heavy computation to **Engines** like Anofox and **Models** like CatBoost/LightGBM/XGBoost.

## 🧠 Role in Architecture
If this were a self-driving car:
*   **TSci** is the navigation software deciding which route to take.
*   **OpenAI** is the co-pilot suggesting optimal strategies.
*   **AnoFox** is the steering/engine executing the turns.
*   **Baseline Models** are the drivetrain (LightGBM, CatBoost, XGBoost).

## Core Agents (All OpenAI-Powered)

### 1. Curator (`curator.py`)
*   **Role**: Data QA & Hygiene.
*   **OpenAI Task**: Classify data quality, recommend cleaning strategies.
*   **Logic**: Computes table metrics via Anofox → asks OpenAI for structured recommendation → returns JSON for `tsci.qa_checks`.
*   **Guardrails**: Never invents data, only reasons over provided metrics.
*   **Output**: `{"data_quality": "pass|warn|fail", "outlier_strategy": "...", "risk_flags": [...]}`

### 2. Planner (`planner.py`)
*   **Role**: Model Selection Strategist.
*   **OpenAI Task**: Suggest model candidates and hyperparameter bands.
*   **Logic**: Given bucket/regime/horizon → OpenAI proposes candidate models → TSci writes jobs to `tsci.jobs`.
*   **Guardrails**: Big 8 are overlays, models see ALL features.
*   **Output**: `{"candidate_models": [...], "hyperparam_ranges": {...}, "focus_features": "all"}`
*   **Key Feature**: `suggest_model_candidates()`, `plan_training_sweep()`

### 3. Forecaster (`forecaster.py`)
*   **Role**: Ensemble Orchestration & Risk Simulation.
*   **OpenAI Task**: Recommend QRA ensemble weights based on regime.
*   **Logic**: Gathers model forecasts → OpenAI suggests weights → executes QRA (L3) → runs Monte Carlo (L4).
*   **Guardrails**: LLM suggests, numeric execution is ours (no black-box ensembling).
*   **Modules Used**:
     - `model_sweep.py` - AutoML-lite sweeps
     - `src/ensemble/qra_ensemble.py` - L3 ensemble
     - `src/simulators/monte_carlo_sim.py` - L4 risk simulation
*   **Output**: Forecast distributions + risk metrics (VaR, CVaR, scenarios)

### 4. Reporter (`reporter.py`)
*   **Role**: Narrative Generation.
*   **OpenAI Task**: Create web-ready reports from forecast results.
*   **Logic**: Retrieves forecasts + bucket contributions + risk metrics → OpenAI generates HTML + JSON.
*   **Guardrails**: Never invents numbers, only explains provided data.
*   **Output**: `{"summary_html": "<p>...</p>", "drivers": [...], "scenarios": [...], "confidence": "high"}`
*   **Dashboard**: Results displayed in `/quant-admin`

## Additional Modules

### 5. Model Sweep (`model_sweep.py`)
*   **Purpose**: Lightweight AutoML per bucket/horizon.
*   **Features**:
     - Trains multiple candidate models
     - Evaluates on validation set
     - Selects best per sweep
     - Logs to `tsci.runs`
*   **Big 8 Role**: Focus tags, NOT filters. Models see all 300+ features.

## OpenAI Integration

### Configuration
```bash
export OPENAI_API_KEY="your_key"
export OPENAI_MODEL="gpt-5.1"  # Or custom: ft:gpt-4.1:org:zl-tsci
```

### Custom Models
- Can use fine-tuned models for TSci-specific reasoning
- Can use RFT (Reinforcement Fine-Tuning) for decision optimization
- See: `docs/ops/OPENAI_CUSTOM_MODEL_GUIDE.md`

### Guardrails
All agents use:
- Structured JSON outputs only
- Hallucination prevention (never invent tables/metrics)
- Temperature 0.1–0.3 (deterministic)
- Safe fallbacks if OpenAI unavailable

## Why is it in `src/models`?
It resides in `models` because TSci contains the **strategic intelligence layer** for our forecasting system. It is not just a utility; it is the agentic "brain" that orchestrates data, features, models, ensembles, and risk—all backed by LLM reasoning.
