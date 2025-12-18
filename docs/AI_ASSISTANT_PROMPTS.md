# AI Assistant Prompt Kit (CBI-V15)

Use these prompts with JetBrains AI Assistant to keep changes aligned with the architecture and guardrails.

## 1) Context & guardrails (paste at session start)

```
You are the CBI-V15 Engineering Agent.

Rules:
- Operate strictly within the CBI-V15 architecture documented in README.md and docs/architecture/MASTER_PLAN.md.
- Ingestion lives under src/ingestion/<source>/.
- Features come from SQL macros in database/macros/ only.
- Training: AutoGluon (Tabular for Big 8, TimeSeries for core ZL), then Meta + Ensemble.
- Forecasts go to MotherDuck; dashboard reads from MotherDuck.
- Never hallucinate directories, files, imports, or dependencies. Ask for missing items.
- Explain your plan before writing code. Produce minimal, surgical diffs.
```

## 2) Code exploration

```
Explain how the ingestion script under src/ingestion/<source>/* works and list the external APIs it calls. Identify env vars needed and failure modes.
```

## 3) SQL macros (AnoFox) assistance

```
Given database/macros/<macro>.sql, suggest testable invariants and write a small SQL smoke test query for CI that validates row counts/ranges and nullability.
```

## 4) Python training tasks

```
Review src/training/* for alignment with AutoGluon presets. Propose a minimal refactor to separate data loading, feature assembly, model training, and evaluation. Output only a concise diff.
```

## 5) Dashboard (Next.js) tasks

```
Given dashboard/, propose a type-safe data access layer for MotherDuck that avoids client-side secrets. Identify which components should be `use server` vs `use client` and provide a minimal example.
```

## 6) Testing prompts

```
Draft unit tests for the most error-prone function in src/simulators/monte_carlo_sim.py. Include edge cases and a quick-run seed for determinism.
```

## 7) Review & hardening

```
Perform an AI Diff Review for this PR and flag: (1) API/contract changes, (2) error handling gaps, (3) missing tests, (4) performance regressions, (5) security/secrets risks.
```

## 8) Runbook & ops

```
Generate a short runbook section for syncing MotherDuck to local DuckDB and launching the dashboard. Include commands and expected artifacts.
```

---

Tips

- Keep outputs short and focused on the files that change.
- Prefer diffs and command blocks over prose.
- If required context is missing, ask targeted questions before proceeding.
