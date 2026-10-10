# HANDOFF

Shared notes between Claude (planning/review) and Junie (implementation).

**Loop:** Claude writes **Current task** → Junie implements and fills **Junie report** → Claude reviews the diff and writes **Review** → next task.

Ownership: Claude edits Current task, Review, Backlog, Decisions. Junie edits Junie report only.

---

## Current task

_None yet. Next up: market baseline (see Backlog)._

---

## Junie report

- **Status:**
- **Files changed:**
- **Tests:**
- **Deviations from plan:**
- **Questions / issues noticed:**

---

## Review

_Claude's review of the last report goes here._

---

## Backlog

Ordered by priority.

1. **Market baseline:** vig-free moneyline probability as a third baseline in `backtest.py`. The real bar the model has to beat.
2. **Remove the vig from moneyline features:** normalize `home_ml_prob` and `away_ml_prob` so they sum to 1.
3. **Drop redundant features:** `ml_prob_diff`, `pf_roll_diff` and similar are exact linear combinations of other columns.
4. **Offseason rest days:** a team's first game of a season gets `rest_days` of about 200. Cap it or add a season-opener flag.
5. **Prior-season baseline in the first season:** with no earlier data loaded, every game falls back to 0.5. Load one extra season for lookup only, or skip those weeks in the baseline.
6. **Pooled metrics:** report metrics over all games alongside the current per-week average.
7. **Ties:** currently counted as away wins. Decide whether to drop them or keep them.

---

## Decisions

- 2026-10-10: Fixed the cross-team `shift(1)` leak in rolling features; added a 72-test suite (branch `fix/rolling-shift-leak`, commit `371c360`).
- 2026-10-10: Workflow: Claude plans and reviews, Junie writes code, coordination through this file.
