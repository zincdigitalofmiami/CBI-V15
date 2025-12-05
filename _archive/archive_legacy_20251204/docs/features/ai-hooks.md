# AI Hooks for TSci Agents

**Date:** December 3, 2024  
**Status:** Implementation complete

---

## Overview

AI hooks enhance TSci agents with OpenAI-powered capabilities:
- **Curator Agent**: AI redundancy detector
- **Planner Agent**: AI experiment designer
- **Reporter Agent**: AI narrative generator

---

## Implementation

**Location:** `src/ai/openai_hooks.py`

**Usage:**
```python
from src.ai.openai_hooks import OpenAIHooks

hooks = OpenAIHooks(api_key="your-key", model="gpt-4")

# Curator: Detect redundancy
redundancy = hooks.detect_redundancy(data_summary)

# Planner: Design experiment
experiment = hooks.design_experiment(features, target, regime, horizon)

# Reporter: Generate narrative
narrative = hooks.generate_narrative(forecast_results, model_performance)
```

---

## Integration Points

### Curator Agent
**File:** `TimeSeriesScientist/time_series_agent/agents/curator_agent.py`

**Add method:**
```python
def detect_redundancy(self, data_summary):
    from src.ai.openai_hooks import OpenAIHooks
    hooks = OpenAIHooks()
    return hooks.detect_redundancy(data_summary)
```

### Planner Agent
**File:** `TimeSeriesScientist/time_series_agent/agents/planner_agent.py`

**Add method:**
```python
def design_experiment(self, features, target, regime, horizon):
    from src.ai.openai_hooks import OpenAIHooks
    hooks = OpenAIHooks()
    return hooks.design_experiment(features, target, regime, horizon)
```

### Reporter Agent
**File:** `TimeSeriesScientist/time_series_agent/agents/reporter_agent.py`

**Add method:**
```python
def generate_narrative(self, forecast_results, model_performance):
    from src.ai.openai_hooks import OpenAIHooks
    hooks = OpenAIHooks()
    return hooks.generate_narrative(forecast_results, model_performance)
```

---

## Configuration

**Environment Variable:**
```bash
export OPENAI_API_KEY="your-api-key"
```

**Model Selection:**
- Default: `gpt-4`
- Alternative: `gpt-3.5-turbo` (faster, cheaper)

---

**Last Updated:** December 3, 2024

