# Junie guidelines for predNFL

Planning happens in a separate Claude chat. The two agents coordinate through `HANDOFF.md` at the repo root.

## Before starting any task
1. Read `HANDOFF.md`. The task to work on is under **Current task**.
2. Only do what that task describes. If something else looks wrong, note it in your report and don't fix it.
3. If the plan is unclear or conflicts with the code, stop and write your question under **Junie report → Questions** instead of guessing.

## While working
- Follow the steps in order. Keep changes inside the listed files unless a step says otherwise.
- Add or update the tests the task names. Run `pytest` before finishing; all tests must pass.
- Don't change existing tests to make them pass unless the task says the behavior is meant to change.

## When finished
Fill in **Junie report** in `HANDOFF.md`:
- Status: `done`, `blocked`, or `partial`
- Files changed
- Test result (the last line of `pytest` output)
- Deviations from the plan, and why
- Questions or issues you noticed but didn't fix

Don't edit the **Current task**, **Review**, **Backlog** or **Decisions** sections; Claude maintains those.

## Project basics
- Source in `src/prednfl/`, tests in `tests/` (offline; data loading is stubbed).
- Install: `pip install -e ".[dev]"`. Test: `pytest`.
- All rolling features must use only games *before* the current one, computed per team (see `_prior_rolling_mean` in `features.py`).
