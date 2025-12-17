# Documentation

> Fast-moving workspace: always read the latest `docs/architecture/MASTER_PLAN.md`, `AGENTS.md`, `DATA_LINKS_MASTER.md`, and the active master plan `.cursor/plans/ALL_PHASES_INDEX.md` before editing docs. Verify current files; avoid duplicating or scattering markdown—keep placement correct.

## Purpose

All project documentation - architecture, operations, feature specs.

## Structure

```
docs/
├── architecture/    # System design, MASTER_PLAN, meta-learning
├── features/        # Feature specs/research
└── ops/             # Operational guides, audits, runbooks
```

## What Belongs Here

- Markdown docs only (architecture, ops, research)
- Diagrams/images supporting docs

## What Does NOT Belong Here

- Code → `src/`
- Config → `config/`
- SQL → `database/`
- Trigger jobs → `trigger/`

## Key Documents (read-first)

- `architecture/MASTER_PLAN.md` — V15.1 source of truth
- `architecture/META_LEARNING_FRAMEWORK.md` — meta-learning/ensemble approach
- `.cursor/plans/ALL_PHASES_INDEX.md` — active master implementation plan (Phases 0–5)

## Hygiene

- Update existing docs; do not add new markdown unless permanent and properly placed.
- Keep filenames and paths aligned with the current repo structure (Trigger-based ingestion, DuckDB/MotherDuck).

## Big 8 Bucket Modeling Rules

Apply these rules in all architecture and planning docs:

- Big 8 buckets are: Crush, China, FX, Fed, Tariff, Biofuel, Energy, Volatility
- Bucket features are derived from SQL macros only in `database/macros/`
- Use AutoGluon `TabularPredictor` for all Big 8 bucket specialists
- Use AutoGluon `TimeSeriesPredictor` for core ZL forecasting
- Meta model fuses Big 8 + core ZL outputs
- Ensemble layer smooths predictions into final forecasts
- Monte Carlo simulation produces probabilistic scenarios (VaR/CVaR), not raw forecasts

## Engineering Agent Prompt (Codex/Cursor)

Use this developer prompt when running Codex/Cursor against this repo:

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
