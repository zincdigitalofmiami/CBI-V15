# AnoFox: The SQL Execution Engine ("The Muscle")

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

## Relationship with TSci
*   **TSci** asks: "Please calculate volatility."
*   **AnoFox** answers: "Here is the result (calculated in 0.01s)."
*   **Separation of Concerns**: AnoFox does *not* decide what to calculate. It just calculates.
