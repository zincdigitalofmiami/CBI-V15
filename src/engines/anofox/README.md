# AnoFox: The SQL Execution Engine ("The Muscle")

> Fast-moving workspace: review `docs/architecture/MASTER_PLAN.md`, `database/macros/`, and the active master plan `.cursor/plans/ALL_PHASES_INDEX.md` before changes. SQL-first; avoid duplicate macros or scripts; keep explorer clean.

AnoFox is the high-performance **Execution Layer**. It is designed to run computations directly where the data lives (MotherDuck / DuckDB) to avoid slow Python loops.

## 🛠 Role in Architecture
*   **Optimization**: It translates high-level requests ("Get me the SMA") into low-level SQL (`AVG(close) OVER (...)`).
*   **Speed**: It processes millions of rows in milliseconds using C++ vectorization (via DuckDB).

## Capabilities

### 1. Unified Bridge (`anofox_bridge.py`)
This is the single entry point. It handles the connection to MotherDuck and loads the necessary extensions.

### 2. Tabular Operations
*   **Gap Filling**: SQL-native interpolation.
*   **Outlier Detection**: Z-Score filtering in SQL.

### 3. Statistical Extensions
*   Computes `RSI`, `MACD`, `Volatility` using window functions.
*   No Pandas lag!

### 4. Forecasting
*   Wraps SQL-based forecasting functions (e.g., `time_bucket()`, `bar()`).

## Relationship with Orchestration Layers
*   **Orchestration layer (e.g., AutoGluon training scripts, legacy AutoGluons)** asks: "Please calculate volatility."
*   **AnoFox** answers: "Here is the result (calculated in 0.01s)."
*   **Separation of Concerns**: AnoFox does *not* decide what to calculate. It just calculates.

## Big 8 Bucket Modeling Rules

When exposing features to the training stack via AnoFox:

- Big 8 buckets are: Crush, China, FX, Fed, Tariff, Biofuel, Energy, Volatility
- Bucket-level features must be implemented as SQL macros in `database/macros/`
- AutoGluon `TabularPredictor` consumes these features for all Big 8 bucket specialists
- AutoGluon `TimeSeriesPredictor` is used for core ZL forecasting (time series)
- Meta model fuses Big 8 + core ZL outputs
- Ensemble layer smooths predictions into final forecasts
- Monte Carlo simulation consumes final forecasts to build probabilistic scenarios (VaR/CVaR)

## Engineering Agent Prompt (Codex/Cursor)

Use this developer prompt when modifying AnoFox engine code with Codex/Cursor:

```text
You are the CBI-V15 Engineering Agent.

Follow the system rules and Cursor rules.json.

Task:
I want you to operate strictly within the CBI-V15 architecture.
Before making any changes:
1. Validate context.
2. If any file or directory is missing, ask me for it.
3. Explain your plan BEFORE writing code.
4. Produce minimal, surgical diffs.

Never hallucinate imports, modules, directories, dependencies, or data sources.
Never reintroduce BigQuery or v14 patterns.
Never write code outside the defined directories.
Keep everything aligned with the V15.1 training engine: Big 8 Tabular → Core TS → Meta → Ensemble → Monte Carlo.

When ready, ask: "Show me the files involved in this operation."
```
