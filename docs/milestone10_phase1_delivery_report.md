# Milestone 10 Phase 1 -- Deterministic Market Data Platform and Feature Store: Delivery Report

## 1. Baseline

Started from commit `7ea2144` ("Integrate portfolio risk authorization
with execution gateway" -- Milestone 9 Phase 4's own delivered baseline).
`HEAD` remains `7ea2144` throughout this phase; nothing has been
committed.

## 2. Scope delivered

A new, self-contained package, `quant_platform.market_data`, implementing
every module the milestone specification named: `identity.py`,
`events.py`, `candles.py`, `ticks.py`, `calendar.py`, `macro.py`,
`quality.py`, `normalization.py`, `feature_store.py`,
`feature_generation.py`, `replay.py`, `verification.py`, `reports.py`,
plus `__init__.py`.

Per the specification's strict scope: no new ML models, no strategy
optimization, no live broker, no MT5, no execution-gateway changes, no
portfolio-risk changes, no prediction logic, no cloud services, no
network streaming, no websocket clients, no credentials, no live
trading, no Milestone 11 work, and (per this instruction) no commit and
no push.

The only file outside the new package that changed is
`src/quant_platform/core/exceptions.py`, and only additively: a new
`MarketDataError` exception tree (13 subclasses) was appended after the
existing `PortfolioRiskLockError` definition. No existing exception
class, docstring, or line was modified or removed.

## 3. Integration design

See `docs/market_data_architecture.md` for the full design. Summary of
the load-bearing decisions:

- **Shared envelope, typed payload.** Every event kind (`Tick`, `Quote`,
  `Trade`, `Candle`) shares one envelope (`event_id`, `instrument_id`,
  `provider`, `symbol`, `event_time`, `arrival_time`, `timeframe`,
  `sequence`, `source_event_id`, `payload`), matching the specification's
  literal field list exactly; kind-specific numeric fields live inside
  `payload` (JSON-safe `Decimal`-as-string), exposed back out via typed
  properties.
- **One-way dependency graph inside the package**, specifically to avoid
  a circular import between `events.py` (which builds the
  `MarketDataEvent` union from `Candle`/`Tick`) and `candles.py`/
  `ticks.py` (which must therefore not depend back on `events.py`).
  Shared field validators were centralized in `identity.py` for this
  reason rather than duplicated per-file (documented in `identity.py`'s
  own module docstring as a deliberate, narrow deviation from this
  repository's usual per-file-duplication convention).
  `MarketEventStore` (raw events) has no separate hash-chain field,
  unlike `portfolio_risk.ledger`'s ledger entries -- every event is
  already content-addressed, so sequence-gaplessness plus no-duplicate-
  id-with-different-content is the complete physical-integrity
  invariant.
  `FeatureStore` is keyed by economic coordinate, not physical position,
  since feature generation may legitimately be re-run over an
  overlapping window and must be idempotent when it reproduces the same
  value.
- **Reuse over duplication.** `compute_content_id` (from
  `paper_trading.identity`), `experiment_lock` (from `ml.concurrency`),
  `canonical_json_bytes`/timestamp formatting (from `core.json`/
  `ml.persistence`), `ValidationIssue`/`ValidationReport`/
  `ValidationSeverity` (from `ml.models`), and -- most substantially --
  the entire `historical.calendar.TradingCalendar` session/holiday/
  maintenance-break model are reused unchanged rather than re-derived.
  Zero lines of `historical`, `features`, `data`, `execution_gateway`, or
  `portfolio_risk` were modified.

## 4. Feature generation catalog

Implemented (all pure, deterministic, `Decimal`-typed): `returns`,
`log_returns`, `rolling_mean`/`sma`, `rolling_std`, `atr`, `rsi`, `ema`,
`vwap`, `price_delta`, `volume_delta`, `high_low_range`, `body_size`,
`upper_wick_ratio`, `lower_wick_ratio` -- 14 named feature series (15
counting `sma`/`rolling_mean` as the same underlying function under two
names), meeting the milestone's "at least" list in full. Disclosed
simplifications (simple- vs Wilder-smoothed ATR/RSI; cumulative vs
session-reset VWAP; sample-vs-population `rolling_std`) are documented in
the architecture doc, chosen deliberately to keep Phase 1's tested
surface to one well-defined variant per indicator.

## 5. Defects found and fixed during this phase's own development

Both were found via this phase's own smoke-testing and unit-test-writing
process, before any test was reported as passing, and both are fixed at
root cause with regression tests now in the suite -- neither was ever
shipped or reported as working.

**Defect #1 -- Windows-incompatible default `instrument_id` separator.**
`normalization.derive_instrument_id`'s first implementation joined
`provider`/`symbol` with `:` (`f"{provider}:{symbol}"`). Both
`MarketEventStore` and `FeatureStore` use `instrument_id` directly as a
filesystem path component; `:` is a reserved drive-separator character on
Windows, and the very first end-to-end smoke test (`create_candle` -> `
MarketEventStore.append`) failed with `NotADirectoryError` on this
platform. Root cause confirmed via direct reproduction of the traceback.
Fixed by changing the separator to `__` (double underscore), which is
safe on every platform this repository targets; a regression test
(`test_market_data_normalization.py::TestDeriveInstrumentId::
test_never_contains_a_colon`) now asserts this explicitly.

**Defect #2 -- structurally unreachable "duplicate ids" quality check.**
`quality.run_candle_quality_checks`'s first implementation detected
"duplicate ids" by comparing each candidate `Candle`'s own `event_id`.
Because `Candle.event_id` bakes in `sequence` (part of its content
identity) and `run_candle_quality_checks` assigns each row its own unique
`sequence=row_index` before construction, no two rows in a single batch
could ever produce a colliding `event_id` -- the check was dead code that
would pass any test asserting only "no crash," never "the check actually
fires." Found while writing
`test_market_data_quality.py::TestDuplicateIds`'s own positive-case test,
which failed with an empty result set instead of the expected finding.
Root-caused to the `sequence`-in-identity interaction described above.
Fixed by redefining the check to compare each row's full `(open_time,
open, high, low, close, volume)` content tuple instead of the
(structurally-guaranteed-unique-at-this-stage) `event_id` -- the row-level
analogue of an id collision, detectable before any real sequence exists.
True post-sequencing event-id collision detection remains
`verification.verify_market_event_store`'s job, and is separately tested
there. Documented in both `quality.py`'s own inline comment and the
architecture doc.

No other defect was found. No test was ever weakened, skipped, or had
its assertions loosened to make it pass.

## 6. Tests

`tests/unit/market_data/` -- 10 files, 1,344 lines, **134 tests**,
covering every category the specification named: normal generation
(`TestGenerateCandleFeaturesDriver`, `TestCleanData`), boundary windows
(`TestRollingMeanAndStd::test_exact_boundary_at_window_minus_one_is_still_none`,
`TestATR`, `TestEMA`), missing data (`TestMissingCandlesAndTimeframeGaps`,
`TestVWAP::test_missing_volume_breaks_the_cumulative_series_from_that_point_on`),
duplicates (`TestDuplicateCandles`, `TestDuplicateIds`,
`MarketEventStore`/`FeatureStore` idempotent-append tests), bad
timestamps (`TestTimestampDisorder`, `TestFutureTimestamps`,
`TestMissingOrInvalidOpenTime`), replay (`test_market_data_replay.py`,
including rebuild-from-raw-events-only and divergence detection), quality
reports (`test_market_data_quality.py`, 20 tests), determinism
(`TestDeterminism`, plus every store/generation test's own
identical-arguments-produce-identical-ids assertions), and
cross-process reproducibility (`TestCrossProcessReproducibility` --
two real, separate `python -c` subprocesses with `PYTHONHASHSEED=0` and
`PYTHONHASHSEED=4294967295`, asserting an identical semantic digest,
mirroring `portfolio_risk.replay`'s own established subprocess-proof
pattern exactly).

Verification's forged-identity and conflicting-history tests hand-corrupt
a real, on-disk `.jsonl` line (via direct file mutation, not a mocked
object) before re-verifying -- proving the checks recompute from raw
bytes rather than trusting any cached/in-memory state.

## 7. Quality gates -- exact results

- `git diff --check`: clean, no output.
- `ruff check .` (full repo): **All checks passed!**
- `mypy src` (full repo, 289 source files): **Success: no issues found.**
- Focused `market_data` suite: `python -m pytest tests/unit/market_data/
  -q` -> **134 passed** (also re-run once with `-W error`: 134 passed,
  zero warnings).
- Determinism repeat: the full 134-test suite was run **10 consecutive
  times**; every run reported **134 passed**, ~17.7s each, with zero
  flakes or variance.
- Additional diligence (not explicitly required by this phase's gate
  list, performed because `market_data` reuses `historical`/
  `paper_trading`/`ml` infrastructure): `tests/unit/historical`,
  `tests/unit/paper_trading`, and `tests/unit/ml` were run together --
  **1849 passed, 1 skipped** (the skip is a pre-existing, unrelated
  Windows-elevated-privileges symlink test, not caused by this phase).
  Confirms the purely-additive `core/exceptions.py` change introduced no
  regression anywhere it is reused.

No focused-test failure occurred at any point in this phase requiring a
root-cause investigation beyond the two defects in Section 5 (both found
and fixed before any test was ever reported as passing).

## 8. Known limitations (see architecture doc for full detail)

No instrument registry (a derived `provider__symbol` id is used); no
tick-to-candle resampling; `vwap` is cumulative, not session-reset;
`atr`/`rsi` use simple, not Wilder, smoothing; macro events are modeled
and point-in-time-safe but not yet wired into feature generation (no
macro-derived feature exists); no CLI surface. All are disclosed, none
are silent gaps.

## 9. Current git status (exact)

```
 M src/quant_platform/core/exceptions.py
?? docs/market_data_architecture.md
?? docs/milestone10_phase1_delivery_report.md
?? src/quant_platform/market_data/
?? tests/unit/market_data/
```

`git diff --cached`: empty (nothing staged). `HEAD`: `7ea2144`,
unchanged from baseline.

## 10. Explicit confirmations

- Nothing has been staged (`git add` was never run this phase).
- Nothing has been committed (`HEAD` is still `7ea2144`).
- Nothing has been pushed.
- Milestone 11 has not been started, and no file outside this phase's
  named scope (the new `market_data` package, its tests, its two docs,
  and the purely-additive `core/exceptions.py` append) was touched.
- No MT5, live broker, network, credential, or live-trading code exists
  anywhere in the new package.

Stopping here per instruction, pending review and explicit commit
approval.
