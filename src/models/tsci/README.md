# TSci: The Intelligence Framework ("The Brain")

TSci is the **System 2 Thinking** layer of the platform. It is a collection of "Agents" that make decisions about *what* to do, but they delegate the actual heavy computation to **Engines** like AnoFox.

## 🧠 Role in Architecture
If this were a self-driving car:
*   **TSci** is the navigation software deciding to turn left.
*   **AnoFox** is the steering mechanism turning the wheels.

## Core Agents

### 1. Curator (`curator.py`)
*   **Role**: Data QA & Hygiene.
*   **Logic**: Checks data health. If it finds gaps, it decides *which* cleaning strategy to use (e.g., "Use Linear Interpolation for small gaps, but delete rows for large gaps").
*   **Action**: Calls `AnoFox.clean_data()`.

### 2. Planner (`planner.py`)
*   **Role**: Feature Engineering Strategist.
*   **Logic**: Decides *which* features matter for the current market regime. 
*   **Example**: "The market is volatile, so we need RSI and Bollinger Bands."
*   **Action**: Calls `AnoFox.calculate_features()`.

### 3. Forecaster (`forecaster.py`)
*   **Role**: Model Selection.
*   **Logic**: Decides *which* model is best right now.
*   **Example**: "We are in a trending market, so use Chronos-2 or Prophet. Do not use Mean Reversion."
*   **Action**: Calls `AnoFox.generate_forecast()` (running the actual math).

### 4. Reporter (`reporter.py`)
*   **Role**: Performance Analyst.
*   **Logic**: Looks at past predictions vs. actuals.
*   **Action**: Updates the "Champion" model registry.

## Why is it in `src/models`?
It resides in `models` because it contains the **domain logic** for our proprietary "Total Social & Technical Intelligence" modeling approach. It is not just a utility; it is the core intellectual property.
