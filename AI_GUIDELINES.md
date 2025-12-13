# AI AGENT MASTER GUIDELINES (CBI-V15)

**Path:** `/Volumes/Satechi Hub/CBI-V15/AI_GUIDELINES.md`

## 1. 🛑 CRITICAL: READ BEFORE ACTING
Before generating code or plans, ingest the current context in this exact order:
1. `docs/architecture/MASTER_PLAN.md` (Strategic intent)
2. `AGENTS.md` (Naming conventions & definitions)
3. `database/README.md` (Data schemas)
4. `AI_GUIDELINES.md` (This file - Operational rules)
5. Active Plan: Check `.cursor/plans/` for the latest active `.plan.md`.

---

## 2. 🧠 OPERATIONAL PROTOCOL (The "No Hallucination" Loop)
You are a Senior Quant Engineer. You do not guess; you verify.

### Phase 1: Verification (Stop & Look)
- **Check First:** Never assume a file exists or an API is available. Confirm paths (e.g., `ls`, `rg`) before referencing.
- **No Magic Imports:** Do not import libraries or local modules without verifying they are installed/exist in the codebase.
- **Context Check:** If asked for a refactor, read the entire target file first.

### Phase 2: Execution (ML & Quant Standards)
- **Quant Finance Rigor:**
  - **No Look-Ahead Bias:** Feature engineering must rely only on past data.
  - **Data Integrity:** Use high-precision types (e.g., `Decimal` for currency). Do not truncate float precision arbitrarily.
  - **Naming:** Follow `AGENTS.md`. Use `volatility_*` or `volume_*` (never `vol_*`).
- **ML Modeling Best Practices:**
  - **Leakage Prevention:** Strict Train/Validation/Test separation. Normalization params fit on Train only.
  - **Reproducibility:** All models accept `random_state` or `seed`; hardcode the seed in configs.
  - **Sanity Checks:** Assert shapes (e.g., `df.shape`) before/after transformations.

### Phase 3: Review (Self-Correction)
- **Double-Check Work:** Review your own diff; avoid deleting critical content.
- **Build Integrity:** After code edits, run relevant tests/linters.
- **Clean Up:** Remove temporary debug prints or `.tmp` files created during the session.

---

## 3. 📂 WORKSPACE & FILE HYGIENE

### Augment & Agent Specifics
- **Config Location:** Keep Augment configs/ignores strictly inside `augment/` (e.g., `augment/.augment.md`).
- **No Data in Configs:** Do not place data, notebooks, or source code under `augment/`.
- **Consistency:** Mirror the structure of `.cursor/README.md` and `.kilocode/README.md`.

### Cleanliness Rules
- **No Root Clutter:** Do not create `temp.py`, `notes.txt`, or stray markdown files in the root. Update existing docs or place new permanent docs under `docs/`.
- **Plan Updates:** If you change the architecture, update the active plan in `.cursor/plans/`.

---

## 4. ✅ DO / ❌ DON'T CHEATSHEET

| Category | ✅ DO | ❌ DON'T |
| :--- | :--- | :--- |
| **Code** | Check for existing utils in `src/shared/` before writing new ones. | Duplicate code that already exists elsewhere. |
| **ML** | Log metrics to specific tracking files/W&B. | Rely on console printouts for model performance. |
| **Files** | Use `augment/` for configs only. | Put code or datasets in the `augment/` folder. |
| **Logic** | Admit if you don't see a file. | Hallucinate a file path just to satisfy the prompt. |
| **Safety** | Backup/Duplicate complex files before large refactors. | Overwrite massive files without verifying the backup. |
