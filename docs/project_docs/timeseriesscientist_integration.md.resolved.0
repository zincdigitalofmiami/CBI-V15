# TimeSeriesScientist (TSci) & Anofox Integration Strategy

## 1. Architectural Relationship
**TSci is the Brain (Orchestrator), Anofox is the Muscle (Engine).**

The user's insight is correct: **TSci must be established first** because it defines the *strategy* for how data is handled, while Anofox provides the *tools* to execute that strategy efficiently within DuckDB.

| Component | Role | Responsibility |
|-----------|------|----------------|
| **TimeSeriesScientist (TSci)** | **Agentic Orchestrator** | Reasoning, decision making, model selection, report generation. It "thinks" about the data. |
| **Anofox** | **SQL-Native Engine** | Heavy lifting, data cleaning, feature calculation, statistical baselines. It "crunches" the data. |

## 2. Integration Points by Agent

### A. Curator Agent (Data Prep)
*   **TSci Role**: Analyzes data statistics (missingness, outliers) and *decides* on a cleaning strategy (e.g., "Use linear interpolation for small gaps, but flag large gaps").
*   **Anofox Role**: Executes the decision using high-performance SQL macros directly in DuckDB.
    *   *Action*: TSci calls `anofox_gap_fill` or `anofox_outlier_detect` instead of writing slow Python loops.

### B. Planner Agent (Model Selection)
*   **TSci Role**: Looks at the "big picture" (policy regime, volatility) to decide if we need a TCN, Transformer, or simple ARIMA.
*   **Anofox Role**: Provides the *inputs* for this decision by calculating `anofox_volatility` and `anofox_trend_strength` features.

### C. Forecaster Agent (Prediction)
*   **TSci Role**: Manages the ensemble. If policy risk is high, it might down-weight the deep learning models and rely more on robust statistical baselines.
*   **Anofox Role**: Generates those statistical baselines (`anofox_arima`, `anofox_ets`) rapidly in-database to serve as stable anchors for the ensemble.

### D. Reporter Agent (Transparency)
*   **TSci Role**: Writes the human-readable explanation: "We switched to a robust ensemble because Anofox detected a regime shift."
*   **Anofox Role**: Provides the raw metrics and logs for the report.

## 3. Setup Order (Critical Path)

1.  **Install TSci Framework**: Set up the agentic environment (Python + LLM keys).
2.  **Define Schema**: TSci's Curator should inspect the raw Databento data and *propose* the optimal DuckDB schema (which we already know should be strict types).
3.  **Initialize Anofox**: Load the extension in DuckDB.
4.  **Ingest Data**: Load ZL futures data into the schema defined in step 2.
5.  **Run Pipeline**: TSci takes over, directing Anofox to process the data.

## 4. Immediate Action Plan
1.  **Clone TSci Repo**: `git clone https://github.com/Y-Research-SBU/TimeSeriesScientist.git`
2.  **Configure Agents**: Set up `config.yaml` to point to our local DuckDB instance.
3.  **Custom Tooling**: Write a wrapper so TSci's Curator can call Anofox SQL functions.
