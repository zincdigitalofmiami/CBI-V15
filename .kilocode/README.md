# Kilocode Guidelines (CBI-V15)

## AI AGENT MASTER GUIDELINES (CBI-V15)

**Path:** `/Volumes/Satechi Hub/CBI-V15/AI_GUIDELINES.md`

Read before tasks:

1. `docs/architecture/MASTER_PLAN.md`
2. `AGENTS.md`
3. `database/README.md`
4. `AI_GUIDELINES.md`
5. Active plan in `.cursor/plans/*.plan.md`

Protocol:

- Verify paths/imports; no hallucination.
- Prevent leakage/look-ahead; train-only stats; set seeds.
- Follow naming rules (`volatility_*` vs `volume_*`; never `vol_*`).
- Keep Kilocode configs only here; avoid root clutter; run sanity checks/tests.

## Purpose

- Keep all Kilocode configs/ignores contained in `.kilocode/`.
- Prevent stray files at repo root after the cleanup.

## File Placement

- `.kilocode/mcp.json` — Kilocode config.
- `.kilocode/.kilocodeignore` — Ignore rules for Kilocode.
- Do **not** add other files under `.kilocode/` unless they are Kilocode-specific configs.
- Never leave Kilocode files at repo root.

## Cleanliness Rules

- No markdown, data, or code in `.kilocode/` — configs only.
- If additional Kilocode settings are needed, place them here and nowhere else.
- Keep the folder minimal; remove stale/temporary files promptly.

## Do / Don’t

- ✅ Store all Kilocode configs inside `.kilocode/`.
- ✅ Keep `.kilocodeignore` inside this folder (not root).
- ❌ Don’t scatter Kilocode artifacts in other directories or repo root.
- ❌ Don’t add non-config content (data, docs, code) to this folder.
