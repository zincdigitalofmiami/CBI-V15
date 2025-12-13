# Cursor Guidelines (CBI-V15)

## AI AGENT MASTER GUIDELINES (CBI-V15)
**Path:** `/Volumes/Satechi Hub/CBI-V15/AI_GUIDELINES.md`

Read before tasks:
1) `docs/architecture/MASTER_PLAN.md`
2) `AGENTS.md`
3) `database/README.md`
4) `AI_GUIDELINES.md`
5) Active plan in `.cursor/plans/*.plan.md`

Protocol:
- Verify paths/imports before use; no hallucination.
- Prevent leakage/look-ahead; train-only stats; set seeds.
- Follow naming rules (`volatility_*` vs `volume_*`; never `vol_*`).
- Run sanity checks/tests; keep logs/plans in `.cursor/`; avoid root clutter.

## Purpose
- Keep Cursor workspace artifacts organized.
- Centralize implementation plans and agent debug output without polluting repo root.

## File Placement
- Plans live in `.cursor/plans/` (e.g., `autogluon_hybrid_implementation_c2287cb0.plan.md`).
- Logs stay in `.cursor/debug.log` (auto-generated).
- Do **not** add other files under `.cursor/` unless they are Cursor-specific (plans, logs, settings).
- Never place `.cursor` files at repo root; keep everything inside `.cursor/`.

## Read First (for planning)
1) `docs/architecture/MASTER_PLAN.md`
2) `AGENTS.md`
3) `database/README.md`
4) `.cursor/plans/autogluon_hybrid_implementation_c2287cb0.plan.md` (active plan)

## Cleanliness Rules
- No stray markdown or config files outside their folders.
- If you create a new plan, save it in `.cursor/plans/` only.
- Delete stale/duplicate plans after completion to keep the list clean.

## Do / Don’t
- ✅ Use `.cursor/plans/` for all plan artifacts.
- ✅ Keep logs confined to `.cursor/debug.log`.
- ❌ Don’t store data, code, or temp notes in `.cursor/`.
- ❌ Don’t duplicate plans; remove legacy entries when superseded.

