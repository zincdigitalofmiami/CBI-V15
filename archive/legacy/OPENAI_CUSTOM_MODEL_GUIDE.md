# OpenAI Custom Model Integration Guide

**Last Updated:** December 7, 2024  
**Purpose:** Guide for using custom/fine-tuned OpenAI models with TSci

---

## 🎯 Overview

TSci agents (Curator, Planner, Forecaster, Reporter) use OpenAI models for:
- Data quality decisions
- Model candidate selection
- Ensemble weighting guidance
- Narrative report generation

You can swap in a **custom or fine-tuned model** without code changes by setting the `OPENAI_MODEL` environment variable.

---

## 🔧 Configuration

### Default Model
```bash
# Uses GPT-5.1 by default
export OPENAI_MODEL="gpt-5.1"
```

### Custom Fine-Tuned Model
```bash
# Use your fine-tuned model ID (CBI-V15 specific)
export OPENAI_MODEL="ft:gpt-4.1:org:zl-cbi-v15-2025-01"
```

### Pro/Flagship Model
```bash
# Use highest-tier reasoning model
export OPENAI_MODEL="o3-pro"
```

---

## 🧠 Fine-Tuning for CBI-V15 Orchestration

### What to Fine-Tune On

Create a fine-tuning dataset with your architecture docs and example decisions:

#### Training Data Examples

**Example 1: Curator Decision**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a quantitative data-quality analyst for a soybean oil futures forecasting system..."
    },
    {
      "role": "user",
      "content": "{\"table_name\": \"raw.databento_ohlcv_daily\", \"summary\": {\"row_count\": 5000, \"null_close\": 12, \"min_date\": \"2020-01-01\", \"max_date\": \"2024-12-07\"}}"
    },
    {
      "role": "assistant",
      "content": "{\"data_quality\": \"pass\", \"outlier_strategy\": \"clip\", \"recommendation\": \"gap_fill_linear\", \"risk_flags\": [\"minor_gaps\"]}"
    }
  ]
}
```

**Example 2: Planner Decision**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a model-selection strategist for commodity futures forecasting..."
    },
    {
      "role": "user",
      "content": "{\"bucket\": \"volatility\", \"horizon\": \"1w\", \"regime\": \"high_volatility\", \"recent_metrics\": {\"lightgbm\": {\"rmse\": 0.5}, \"catboost\": {\"rmse\": 0.45}}}"
    },
    {
      "role": "assistant",
      "content": "{\"candidate_models\": [\"catboost\", \"tft\", \"garch\"], \"hyperparam_ranges\": {\"depth\": [4, 8], \"lr\": [0.01, 0.05]}, \"focus_features\": \"all\"}"
    }
  ]
}
```

#### Documents to Include as Context

Include these in your fine-tuning dataset or as few-shot examples:

1. `docs/architecture/MASTER_PLAN.md` - Architecture overview
2. `docs/architecture/META_LEARNING_FRAMEWORK.md` - Model selection logic (historical)
3. `docs/architecture/ENSEMBLE_ARCHITECTURE_PROPOSAL.md` - Ensemble design (historical)
4. `docs/ops/BIG_8_BUCKETS_REFERENCE.md` - Bucket definitions
5. `docs/ops/NAMING_CONVENTIONS.md` - Volatility vs volume rules

---

## 🚀 Fine-Tuning Process

### Step 1: Prepare Training Data

```python
# Create fine-tuning dataset
import json

training_examples = []

# Add architecture docs as system context
with open("docs/architecture/MASTER_PLAN.md") as f:
    arch_doc = f.read()

# Add example decisions (curator, planner, forecaster, reporter)
# ... (see examples above)

with open("cbi_v15_finetuning_data.jsonl", "w") as f:
    for example in training_examples:
        f.write(json.dumps(example) + "\n")
```

### Step 2: Upload and Fine-Tune

```python
from openai import OpenAI

client = OpenAI()

# Upload training file
file = client.files.create(
    file=open("cbi_v15_finetuning_data.jsonl", "rb"),
    purpose="fine-tune"
)

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4.1",  # or "gpt-5.1"
    suffix="zl-cbi-v15-2025-01",
)

print(f"Fine-tuning job created: {job.id}")
print(f"Custom model will be: ft:gpt-4.1:org:zl-cbi-v15-2025-01")
```

### Step 3: Use Custom Model

```bash
# After fine-tuning completes
export OPENAI_MODEL="ft:gpt-4.1:org:zl-cbi-v15-2025-01"

# Run your orchestration script with custom model
python path/to/your_orchestration_planner.py
```

---

## 🎓 Reinforcement Fine-Tuning (RFT)

For even stronger performance, use OpenAI's RFT after supervised fine-tuning:

### Step 1: Define Grader Function

```python
def orchestration_decision_grader(sample: dict, item: dict) -> float:
    """
    Grade orchestration decisions based on downstream forecast performance.
    """
    # Compare suggested models to gold standard
    suggested_models = set(sample.get("candidate_models", []))
    gold_models = set(item.get("best_models", []))
    
    # Jaccard similarity
    if not suggested_models or not gold_models:
        return 0.0
    
    intersection = len(suggested_models & gold_models)
    union = len(suggested_models | gold_models)
    
    return intersection / union if union > 0 else 0.0
```

### Step 2: Run RFT Job

```python
from openai import OpenAI

client = OpenAI()

# Upload RFT training file (with grader results)
file = client.files.create(
    file=open("cbi_v15_rft_data.jsonl", "rb"),
    purpose="fine-tune"
)

# Create RFT job
rft_job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="ft:gpt-4.1:org:zl-cbi-v15-2025-01",  # Your base fine-tuned model
    method="rft",  # Reinforcement fine-tuning
    suffix="zl-cbi-v15-rft",
)

print(f"RFT job: {rft_job.id}")
```

---

## 📊 Model Comparison

| Model Type | Use Case | Latency | Cost | Reasoning Quality |
|------------|----------|---------|------|-------------------|
| **gpt-5.1** | Default, fast decisions | Low | $ | Good |
| **o3-pro** | Complex regime analysis | High | $$$ | Excellent |
| **ft:gpt-4.1:...:zl-cbi-v15** | Domain-specific CBI-V15 | Medium | $$ | Very Good (specialized) |
| **ft:...:zl-cbi-v15-rft** | RFT-optimized CBI-V15 | Medium | $$ | Excellent (tuned to rewards) |

### Recommendation

1. **Start with:** `gpt-5.1` (default, good enough for most cases)
2. **Upgrade to:** Fine-tuned model after you have 50-100 example decisions
3. **Ultimate:** RFT model after you have a validated grader and performance data

---

## ✅ Current Implementation

All TSci agents already support `OPENAI_MODEL`:

```python
# In src/utils/openai_client.py
def get_default_model(model: Optional[str] = None) -> str:
    """Get the default model name (env override, otherwise gpt-5.1)."""
    return model or os.getenv("OPENAI_MODEL", "gpt-5.1")
```

**No code changes needed** - just set the environment variable.

---

## 🔗 References

- OpenAI Fine-Tuning: https://platform.openai.com/docs/guides/fine-tuning
- RFT Cookbook: https://cookbook.openai.com/examples/reinforcement_fine_tuning
- Agent Evaluation: https://cookbook.openai.com/examples/agents_sdk/evaluate_agents
- Hallucination Guardrails: https://cookbook.openai.com/examples/developing_hallucination_guardrails

---

## 🎯 Quick Start

```bash
# 1. Use default model
export OPENAI_API_KEY="your_key"
python path/to/your_orchestration_planner.py

# 2. Use Pro model for complex reasoning
export OPENAI_MODEL="o3-pro"
python path/to/your_orchestration_planner.py

# 3. Use your custom fine-tuned model
export OPENAI_MODEL="ft:gpt-4.1:org:zl-cbi-v15-2025-01"
python path/to/your_orchestration_planner.py
```
