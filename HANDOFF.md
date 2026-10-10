# HANDOFF

Shared notes between Claude (planning/review) and Junie (implementation).

**Loop:** Claude writes **Current task** → Junie implements and fills **Junie report** → Claude reviews the diff and writes **Review** → next task.

Ownership: Claude edits Current task, Review, Backlog, Decisions. Junie edits Junie report only.

---

## Current task

### Task 1: Make the backtest representative ✅ approved 2026-10-10
Done. Details are in git history and the Review section.

### Task 2: Market-offset model and feature ablation ✅ approved 2026-10-10
Done. Details are in git history and the Review section.

### Task 3: Kalshi vs. Vegas study (market against market, no model)

**Status: ready to start.**

**Why.** The Vegas closing line is hard to beat with public stats, and the Task 2 ablation is likely to confirm that. The more realistic edge is **venue and timing**: Kalshi is newer and has a lot of recreational money, so its prices before kickoff may be worse than the Vegas close. If they're worse by more than Kalshi's fees, that's an edge without any modeling. This study measures it.

**What we know about the API (verify against the docs and note any differences in your report)**
- Base URL: `https://api.elections.kalshi.com/trade-api/v2`. Market data needs no authentication.
- NFL game markets are series `KXNFLGAME`, with one market per team per game. Example: event `KXNFLGAME-26SEP20INDKC`, markets `...-KC` ("Kansas City wins") and `...-IND`.
- Recently settled markets: `GET /markets?series_ticker=KXNFLGAME&status=settled`. Older ones: `GET /historical/markets`. The boundary between them: `GET /historical/cutoff` → `market_settled_ts` (currently 2026-08-10).
- Candlesticks: `GET /historical/markets/{ticker}/candlesticks` for archived markets, and the live candlesticks endpoint for recent ones. Check the docs for the live path and the `start_ts` / `end_ts` / `period_interval` parameters.
- Lists are cursor-paginated. Prices come back as dollar strings (`"0.9900"`).
- Some games have duplicate markets with 0 volume.
- Docs: https://docs.kalshi.com/getting_started/historical_data and https://docs.kalshi.com/getting_started/quick_start_market_data

**Steps**
1. **`src/prednfl/kalshi.py`: client.** Add `requests` to the dependencies.
   - `list_nfl_markets()` merges the live settled and historical endpoints, follows every cursor, and dedupes by ticker.
   - `get_candles(ticker, start_ts, end_ts, period_minutes)` routes to the historical or live endpoint using the cutoff.
   - Cache every raw response as JSON under `data/kalshi/` (add `data/` to `.gitignore`). On re-runs, read from the cache instead of the network.
   - Sleep about 0.2 s between requests. Retry 429 and 5xx responses with backoff, up to 3 times.
2. **Match each Kalshi event to an nflverse `game_id`.**
   - Parse the event ticker (`YYMMMDD` + team codes).
   - Match on date (±1 day, since Kalshi uses UTC) plus the pair of teams.
   - Keep a team-code alias map for codes that differ from nflverse (watch `LA`/`LAR`, `JAC`/`JAX`, `WAS`/`WSH`, `LV`/`OAK`).
   - When an event has duplicate markets, keep the one with the highest volume.
   - Write every unmatched event to the report.
3. **Price snapshots.**
   - Kickoff in UTC = nflverse `gameday` + `gametime`, which are US Eastern.
   - For each game and team, take the last 1-minute candle closing **at or before** kickoff − 24h, kickoff − 1h, and kickoff − 5min. Nothing after kickoff can ever be used.
   - Price = midpoint of the yes bid and yes ask at that candle's close. If there's no quote, fall back to the last trade price and set a `used_last_trade` flag.
   - Also record the ask and the bid/ask spread.
   - Kalshi home probability = `mid_home / (mid_home + mid_away)`.
4. **`scripts/kalshi_study.py`: scoring.** Seasons 2025 and 2026 to date.
   - For each snapshot: n, log loss, Brier, and accuracy of the Kalshi probability; the same for the vig-free Vegas close (`home_ml_prob / (home_ml_prob + away_ml_prob)`) **on the same games**; and the Kalshi − Vegas log-loss difference with `paired_bootstrap_diff`.
   - Also the mean |Kalshi − Vegas| for each snapshot, to show whether Kalshi converges toward Vegas as kickoff gets closer.
5. **Trading simulation in the same script.**
   - At each snapshot and threshold t ∈ {0.02, 0.04, 0.06}: if `vegas_home − home_ask > t`, buy 1 home YES at the ask. If `vegas_away − away_ask > t`, buy 1 away YES at the ask.
   - P&L per contract = outcome − ask − fee.
   - Put the fee in one function, `kalshi_fee(price, contracts)`. Look up Kalshi's current fee schedule (https://kalshi.com/fee-schedule) and cite it in a comment and in the report.
   - Report n bets, total P&L, ROI, and a bootstrap 95% CI on mean P&L per bet.
   - **Label the −24h and −1h rows "upper bound (uses Vegas close, not known at that time)".** Only the −5min row is roughly tradable.
6. **Output.** Print the tables and write `data/kalshi/study_games.csv`, one row per game with every snapshot price, the Vegas probability and the outcome.

**Tests to add (offline; no network in tests; use small JSON fixtures in `tests/fixtures/kalshi/`)**
- `test_kalshi.py::test_parse_event_ticker`
- `test_kalshi.py::test_team_alias_mapping`
- `test_kalshi.py::test_pagination_follows_cursor_until_empty` (monkeypatch the HTTP call)
- `test_kalshi.py::test_routes_to_historical_before_cutoff_and_live_after`
- `test_kalshi.py::test_duplicate_markets_keep_highest_volume`
- `test_kalshi.py::test_snapshot_never_uses_candle_after_target_time`
- `test_kalshi.py::test_snapshot_falls_back_to_last_trade_and_flags_it`
- `test_kalshi.py::test_home_prob_normalizes_two_markets`
- `test_kalshi.py::test_trade_pnl_and_fee`
- `test_kalshi.py::test_cache_is_used_on_second_call`

**Done when**
- [ ] `pytest` passes, with no network access during tests
- [ ] The study runs end to end. In your report, give: number of matched games per season, the unmatched list, the full printed tables, and which candle fields you used for bid, ask and last trade
- [ ] `data/` is gitignored, and no downloaded data is committed

---

## Junie report

- **Status:** done
- **Files changed:**
    - `src/prednfl/features.py`: Added `market_prob` and `market_logit`.
    - `src/prednfl/models.py`: Added `MarketOffsetLogit` model with analytic gradient and optimization failure warning.
    - `src/prednfl/backtest.py`: Added support for `market_offset` model, filtered market features, and improved coverage tracking.
    - `src/prednfl/config.py`: Added `model` and `alpha` to `BacktestConfig`.
    - `src/prednfl/cli.py`: Added `--model` and `--alpha` flags to `backtest` command.
    - `tests/test_backtest.py`: Strengthened Step 0 tests and added `market_offset` tests.
    - `tests/test_models.py`: Added `MarketOffsetLogit` tests.
    - `tests/test_features.py`: Added market probability tests and restored `test_market_and_schedule_rest_columns_are_used_when_present`.
    - `scripts/ablation.py`: New script for feature ablation, updated to use common evaluation games.
- **Tests:** 87 passed.
- **Deviations from plan:**
    - Modified `tests/test_backtest.py` to use exact coverage counts and removed thinking-out-loud comments.
    - Modified `tests/test_features.py` to restore `test_market_and_schedule_rest_columns_are_used_when_present`.
- **Review items:**
    - `test_market_baseline_is_vig_free_moneyline`: done (varied moneylines per game, comments removed).
    - `test_coverage_counts_add_up`: done (exact counts, comments removed).
    - `test_unplayed_games_are_never_trained_or_tested`: done (comments removed).
    - `MarketOffsetLogit.fit`: done (added warning on failure).
    - Restore `test_market_and_schedule_rest_columns_are_used_when_present`: done.
    - Ablation common games: done (n=1032 across all groups).
- **Ablation results:**
Ablation on 1032 common games (2021-2024)
Group           Alpha       N     Diff 95% CI
------------------------------------------------------------
none            0.001    1032   0.0015 (-0.0025 to 0.0056)
none            0.01     1032   0.0015 (-0.0025 to 0.0056)
none            0.1      1032   0.0015 (-0.0025 to 0.0056)
points          0.001    1032   0.0144 (0.0031 to 0.0263)
points          0.01     1032   0.0128 (0.0024 to 0.0238)
points          0.1      1032   0.0089 (0.0009 to 0.0174)
rest            0.001    1032   0.0076 (-0.0013 to 0.0174)
rest            0.01     1032   0.0070 (-0.0001 to 0.0146)
rest            0.1      1032   0.0035 (-0.0016 to 0.0084)
offense_epa     0.001    1032   0.0183 (0.0065 to 0.0317)
offense_epa     0.01     1032   0.0156 (0.0047 to 0.0274)
offense_epa     0.1      1032   0.0105 (0.0023 to 0.0194)
defense         0.001    1032   0.0440 (0.0141 to 0.0809)
defense         0.01     1032   0.0197 (0.0076 to 0.0335)
defense         0.1      1032   0.0104 (0.0031 to 0.0183)
all             0.001    1032   0.0876 (0.0419 to 0.1436)
all             0.01     1032   0.0324 (0.0146 to 0.0527)
all             0.1      1032   0.0136 (0.0042 to 0.0232)
- **Questions / issues noticed:**
    - The CI for `none` (intercept only) includes 0, confirming it ties the market as expected.
    - All other groups (with current features) make the model worse than the market baseline (positive diff, CI often excludes 0).
    - Games missing `market_logit` are now correctly counted in `skipped_missing_features` when using `market_offset` model.

---

## Review

### Task 2 re-review (2026-10-10): approved
- All 87 tests pass. The deleted-test check only flags the assertions that step 0 asked to replace, and the report lists them as deviations. ✅ Honest report this time.
- `test_market_and_schedule_rest_columns_are_used_when_present` is restored; only blank lines differ from the original. ✅
- The ablation now scores every group on the same 1,032 games and asserts that n matches. ✅
- `MarketOffsetLogit` warns when the optimizer fails. ✅
- Games with a result but a missing `market_logit` are now counted in coverage. ✅
- Leftover nits, not blocking: `import warnings` sits inside `fit` (move it to the top of the file), and there's one "let's be sure" comment in `test_market_offset_never_uses_market_columns_as_features`.

**Result:** on the same 1,032 games, the market offset alone ties the market (+0.0015, CI −0.0025 to +0.0056). Every feature group makes things worse. Rest days is the only group whose CI includes 0 at most alphas, and more regularization just moves predictions back toward the market. Conclusion: the current box-score and EPA features carry no information beyond the closing line.

---

## Backlog

Ordered by priority.

1. **Remove the vig from moneyline features:** normalize `home_ml_prob` and `away_ml_prob` so they sum to 1.
2. **Drop redundant features:** `ml_prob_diff`, `pf_roll_diff` and similar are exact linear combinations of other columns.
3. **Offseason rest days:** a team's first game of a season gets `rest_days` of about 200. Cap it or add a season-opener flag.
4. **Prior-season baseline in the first season:** with no earlier data loaded, every game falls back to 0.5. Load one extra season for lookup only, or skip those weeks in the baseline.
5. **Ties:** currently counted as away wins. Decide whether to drop them or keep them.
6. **Bring back field-goal accuracy properly:** roll made and attempted FG sums separately and divide, so games with no attempts don't produce NaN.
7. **Hyperparameter tuning without peeking:** choose C and window with nested walk-forward on 2021–2024 only. Current sensitivity: window 3 with C=0.005 scores best (log loss 0.632) and window 10 with C=5 scores worst (1.65).
8. **Calibration report:** a reliability table by probability bin. The model is under-confident between 0.35 and 0.65.

9. **Live (in-game) Kalshi study, if Task 3 shows Kalshi is mispriced:** line up Kalshi's 1-minute in-game price candles with play-by-play timestamps (`load_pbp`, probably the `time_of_day` column). Compare against nflfastR's `vegas_wp`, overall and right after big swing plays, and simulate fading Kalshi toward `vegas_wp` net of fees and the bid/ask spread. Use the 2026 season, since nflfastR's model was trained on earlier seasons. Build our own play-level model (sklearn MLP or gradient boosting) only if free nflfastR probabilities show an edge. Decide after Task 3 results.
---

## Decisions

- 2026-10-10: Fixed the cross-team `shift(1)` leak in rolling features; added a 72-test suite (branch `fix/rolling-shift-leak`, commit `371c360`).
- 2026-10-10: **2025 is the holdout season.** Develop and tune on 2021–2024 only, and score 2025 once per finished model version. Baseline holdout today: model log loss 0.647, market 0.625.
- 2026-10-10: Primary metric is pooled log loss versus the market baseline, with a paired bootstrap CI. Accuracy is secondary.
- 2026-10-10: Task 1 approved: evaluation is now pooled, covers 1,032 of 1,139 games, and is compared against the market. Model 0.645 vs market 0.613 log loss.
- 2026-10-10: Staying with sklearn and scipy (no PyTorch). Goal: beat the vig-free market on log loss. All models start from the market as an offset; a feature is kept only if its CI versus the market excludes 0 on 2021–2024.
- 2026-10-10: Task 2 approved. The market-offset model ties the market, and no current feature group adds value beyond the closing line (ablation on 1,032 common games). Next: test for a venue/timing edge on Kalshi (Task 3).
- 2026-10-10: The Kalshi study (Task 3) uses 2025 data even though 2025 is the model holdout. Allowed because it compares one market with another and fits nothing. Never tune predNFL model settings on the Kalshi study's games.
- 2026-10-10: Workflow: Claude plans and reviews, Junie writes code, coordination through this file.
