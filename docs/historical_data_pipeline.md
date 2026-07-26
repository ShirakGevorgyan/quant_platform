# Historical Data Pipeline (Milestone 2)

This is the authoritative technical reference for `quant_platform.historical`
and `quant_platform.data_cli` -- the historical market-data ingestion,
normalization, validation, storage, resampling, versioning, and reproducible
loading pipeline built on top of Milestone 1's backtesting engine. The
README gives the one-paragraph overview; this document gives the contract
each stage makes and the reasoning behind it.

**This layer produces data. It does not produce trading edge.** Everything
below is about making sure the OHLCV bars fed into `TimeframeCursor` and
`BacktestEngine` are exactly what the broker actually reported, correctly
timestamped, and never leaked from the future. It says nothing about
whether a strategy built on top of this data is profitable -- backtest
quality still depends entirely on the underlying broker feed's fidelity
(spread history, requote behavior, out-of-hours quotes), the correctness of
the timezone/session configuration a user supplies for their specific
broker, and everything Milestone 1 already covers (cost modeling,
execution assumptions, no forward-looking bias in the strategy itself).
**A leak-free, checksum-verified dataset is a necessary, not sufficient,
condition for a trustworthy backtest.**

## Data lifecycle

```
HistoricalSource (broker-agnostic)         historical/source.py
   |  MT5HistoricalSource (lazy-imports MetaTrader5)   historical/mt5_adapter.py
   v
Raw immutable snapshot                     historical/raw_store.py
   |  (exactly what the source returned, post schema/dtype normalization,
   |   pre validation/repair -- content-checksummed, atomically written,
   |   never overwritten)
   v
Quality validation                         historical/quality.py
   |  (typed QualityReport: severity, issue type, affected rows -- never
   |   raises on its own; a policy decision, not a detection concern)
   v
Repair / quarantine policy                 historical/repair.py
   |  (STRICT / WARN_ONLY / QUARANTINE; exact-duplicate removal always
   |   permitted; every action recorded in a RepairLineage; no price is
   |   ever interpolated or fabricated)
   v
Canonical Parquet storage (per timeframe)  historical/canonical_store.py
   |  (content-addressed: each year-partition's data is written once per
   |   distinct content checksum under content/<sha256>/, immutable and
   |   never overwritten; a separate CURRENT pointer file names the
   |   casually-current content id -- see "Content-addressed storage")
   v
Leak-free resampling (M1 -> M5/M15/M30/H1/H4/H12/D1)   historical/resampling.py
   |  (epoch-aligned buckets; a derived bar is "complete" only once its
   |   full window has closed within the source data's own coverage)
   v
Incremental update pipeline                historical/update_pipeline.py
   |  (idempotent; overlap-window reconciliation; conflicting revisions
   |   rejected unless explicitly accepted; whole operation held under a
   |   per-dataset advisory lock -- see "Concurrency, locking, and crash
   |   safety")
   v
Dataset manifest                           historical/manifest.py
   |  (immutable, versioned -- version is NEVER just a date -- machine-
   |   readable AND human-inspectable; binds each partition-year to the
   |   EXACT content id that produced it, which is what makes exact
   |   historical reconstruction possible)
   v
Reproducible dataset loader                historical/loader.py
   |  (any manifest version -- current or historical -- reconstructs its
   |   own exact bytes via its recorded content ids)
   v
   -> multiframe.cursor.TimeframeCursor / engine.backtest_engine.BacktestEngine
```

Every arrow above is a real module boundary, not just a conceptual one:
each stage is independently testable and independently useful (e.g. you can
run quality checks against data that never came from this pipeline at all).

## Raw vs. canonical vs. derived

- **Raw** (`historical/raw_store.py`): exactly what the source said, once
  dtype-normalized (see below) but never cleaned, sorted, deduplicated, or
  validated. A raw snapshot is permanent and immutable -- if a downstream
  bug is ever found, every later stage can be re-run from the untouched
  raw snapshots. Layout: `<root>/raw/source=<s>/broker=<b>/symbol=<sym>/
  timeframe=<tf>/snapshot=<id>/{data.parquet,metadata.json,_SUCCESS}`.
- **Canonical** (`historical/canonical_store.py`): validated, repaired,
  deduplicated bars at their SOURCE timeframe (e.g. M1), one content-
  addressed Parquet blob per UTC calendar year per distinct content
  checksum. This is what `apply_incremental_update` maintains and what the
  loader reads by default.
- **Derived** (also `canonical_store.py`, same layout, different
  `timeframe=`): the output of `resample_ohlcv`, stored the same way as
  canonical data but with `DatasetManifest.resampling_config` populated so
  its provenance (source timeframe, completion policy) is always visible.
  Derived and canonical data live in the same store, distinguished only by
  the manifest -- never silently mixed, since every read is scoped to one
  explicit `(symbol, timeframe)` pair.

## Content-addressed storage and exact version reconstruction

This is the property that makes "reproduce a backtest byte-for-byte months
after the dataset has since been revised" actually true, not just aspired
to. Each partition-year's layout:

```
<root>/canonical/symbol=<s>/timeframe=<tf>/year=<Y>/
    content/<sha256>/{data.parquet, metadata.json, _SUCCESS}   # immutable, write-once
    CURRENT                                                      # one line: the current content id
```

- `write_partition` computes the new data's content checksum, then either
  discovers an identical blob already exists (a content-duplicate rewrite
  is a dedup no-op -- "already stored (dedup)") or atomically renames a
  temp directory into `content/<checksum>/`. **An existing `content/<id>/`
  directory is never overwritten, edited, or deleted** -- this is the
  entire guarantee. Only after the blob is durably in place does
  `_set_current` atomically flip the `CURRENT` pointer (temp-file +
  `os.replace`) to reference it.
- `DatasetManifest.partition_content_ids: dict[int, str]` records, for
  EVERY year a dataset's build touched, the exact content id bound at that
  moment -- not "the current one," a specific immutable one. Saving a
  manifest is what freezes a dataset version's exact recipe.
- `DatasetLoader` reconstructs a requested manifest version -- current or
  historical -- entirely from its own recorded `partition_content_ids`, via
  `CanonicalStore.read_partition_by_content_id`, which reads a specific
  content-addressed blob directly and **never consults `CURRENT`**. A
  dataset can be revised arbitrarily many times after a given manifest
  version was recorded; that version's `partition_content_ids` still point
  at the exact original blobs (never deleted, never mutated), so loading it
  again reproduces byte-identical data. This is proven end-to-end by
  `tests/integration/test_reproducibility.py`
  (`TestV1V2ReproducibilityProof`): build V1, fingerprint a backtest run
  against it, revise a historical bar producing V2, reload V1 explicitly,
  and assert the reloaded data (`pandas.testing.assert_frame_equal`) and
  the backtest fingerprint are byte-identical to the original -- while V2
  is provably distinct and both V1 and V2 remain independently loadable.
- Casual reads (no explicit `dataset_version` pinned) resolve through
  `CURRENT`, so day-to-day ingestion/loading pays no extra cost or
  complexity for this guarantee -- it only matters when a caller pins a
  specific version, which is exactly the "reproduce months later" use case.
- This also gives copy-on-write deduplication for free: re-ingesting a
  year whose data hasn't actually changed (the common case for a daily
  incremental update touching only the most recent partition) writes zero
  new bytes, it just confirms the existing blob and (if needed) leaves
  `CURRENT` unchanged.

## Timezone contract (critical)

See `historical/timezones.py`'s module docstring for the full rationale;
the enforced rules, summarized:

1. Every canonical timestamp is tz-aware UTC. Nothing past
   `localize_broker_timestamps` ever sees a naive timestamp.
2. A naive timestamp is **never** silently assumed to be UTC (unlike
   `core.time_utils.ensure_utc`, which deliberately does that for
   already-UTC internal/synthetic data -- a different function for a
   different, narrower purpose).
3. The source timezone is always an explicit, configured value: a fixed
   UTC offset (`FixedOffsetTimezone`, the common case for MT5 "server
   time") or a standards-based IANA zone via `zoneinfo`
   (`NamedZoneTimezone`). Nothing is ever inferred from the host machine's
   local timezone.
4. DST-ambiguous (fall-back) and DST-nonexistent (spring-forward)
   wall-clock times are **rejected**, not guessed -- this pipeline has no
   metadata from a broker to disambiguate them.
5. The MT5 adapter treats `copy_rates_range`'s `time` field (and the
   `date_from`/`date_to` it's given) as naive broker-server wall clock,
   converted through the configured `server_timezone` -- never as literal
   UTC despite the epoch-seconds shape.

On Windows, `NamedZoneTimezone` requires the `tzdata` package (already a
dependency; `zoneinfo` has no bundled database on Windows).

## XAUUSD session/calendar configuration

XAUUSD is not a 24/7 instrument. `historical/calendar.py`'s
`TradingCalendar` models a weekly session window, optional daily
maintenance breaks, and optional holiday closures, all expressed in one
explicit local timezone. `default_xauusd_calendar` is an **illustrative**
default (Sun 23:00 -> Fri 23:00 server time, matching a common retail-
broker convention) -- it is not universal. Configure
`SessionCalendarConfig` in your ingestion config with your actual broker's
published schedule; the calendar is what lets `run_quality_checks`
distinguish "the market was expectedly closed" (an `INFO`-level gap) from
"a bar is missing that shouldn't be" (a `WARNING`).

**Known limitation**: this is a small, explicit model (weekly window +
daily breaks + ad-hoc holidays), not a general exchange-calendar library.
It is sufficient for a spot/CFD instrument's actual closure pattern; it
would need to be extended (or replaced with a proper exchange-calendar
dependency) for an exchange-traded instrument with contract rolls, product-
specific holidays, or half-days.

## Validation severity model

`historical/quality.py::run_quality_checks` never raises -- it returns a
`QualityReport` of typed `QualityIssue`s, each with a `Severity`
(`INFO`/`WARNING`/`CRITICAL`), an `IssueType`, a human-readable message, a
capped display sample of affected timestamps, and the COMPLETE (uncapped)
set of affected row positions (`affected_row_indices`) that
`historical.repair` uses to act precisely. Checks run in three tiers:

1. Per-value sanity (schema, dtypes, nulls, non-finite, non-positive
   price, OHLC invariants, volume/spread sign) -- always run.
2. Temporal shape (ordering, duplicates, grid alignment, gaps, overlaps,
   closed-session bars) -- run together, deliberately not gated on each
   other (a malformed series routinely trips several simultaneously; see
   the code comment in `run_quality_checks` for the specific bug this
   avoided). Gated only on `open_time` itself being usable (tz-aware, no
   nulls).
3. Market-quality heuristics (frozen sequences, impossible jumps, extreme
   spread/range, volume spikes, incomplete edge bars, batch-boundary
   artifacts) -- gated on tiers 1+2 being critical-issue-free, since these
   are statistics computed across rows.

## Repair/quarantine policy

`historical/repair.py::apply_repair_policy` never interpolates a price.
The only actions it can take:

- Remove byte/field-identical duplicate rows (always permitted, zero
  information loss).
- Sort by `open_time` -- only if `allow_sort=True`, and always reported.
- Apply a `SeverityPolicy` to rows carrying a CRITICAL issue:
  `STRICT` (reject the whole batch, the default), `WARN_ONLY` (retain
  everything, note it), or `QUARANTINE` (remove exactly the affected rows
  into a separate, still-inspectable frame).

Every action produces a `RepairStep` in the returned `RepairLineage`
(transformation name/version, parameters, rows removed/changed, affected
row indices) plus a content checksum of the result -- "never hide a
repair" is enforced structurally: there is no code path that changes a row
without recording it.

## Dataset manifests

`historical/manifest.py::DatasetManifest` is the machine-readable AND
human-inspectable record of one dataset version: dataset ID (stable across
versions), version (see below), parent raw-snapshot IDs, symbol/source/
broker/timeframe, UTC date range, row count, schema fingerprint, content
checksum, **per-year content ids (`partition_content_ids`, see "Content-
addressed storage" above)**, pipeline version, **code revision (see below)**,
**reproducibility seed**, normalization/validation/repair summaries,
calendar version, resampling config (for derived datasets), and creation
timestamp. A manifest's `version` is **never just a date** -- it's
`{monotonic sequence}-{content checksum prefix}`, so two different builds
on the same day can never collide or be confused, and saving an exact
content-duplicate of the latest version is a no-op (no redundant version
minted) -- this is what makes incremental updates that legitimately
changed nothing idempotent at the manifest layer.

### Code revision without requiring a Git commit

`historical/code_revision.py::capture_code_revision` records what pipeline
code produced a manifest, without assuming the repository is even under
Git or has a single commit: it tries `git rev-parse HEAD` first
(`git:<hash>`), and if that's unavailable (no Git installed, not a
repository, zero commits) falls back to a deterministic sha256 over every
`.py` file in the `historical` package, sorted by path (`content:<hash>`).
Either way, `apply_incremental_update` records a real, reproducible
provenance value on every manifest -- `code_revision` is never null and
never a placeholder.

## Incremental update semantics

`historical/update_pipeline.py::apply_incremental_update` never blindly
appends. `determine_update_start` computes a small overlapping re-fetch
window (latest canonical bar minus N bar-widths) so a re-ingestion can
detect source-side revisions. Every bar in the overlap is classified as
unchanged (byte-identical), a genuine insert, or a conflict (same
`open_time`, different OHLCV). Conflicts are rejected by default
(`RevisionPolicy.REJECT_CONFLICTS`); accepting a revision
(`ACCEPT_NEWER_SOURCE`) is an explicit opt-in, and every replacement is
still counted in the returned `UpdateReport`. Each year-partition's content
blob is written immutably (never overwritten, per "Content-addressed
storage" above) before `CURRENT` is atomically flipped; a crash between
partitions in a multi-year update leaves the already-written ones valid and
the update safely re-runnable (idempotent, per above) -- this is a
deliberate scale-appropriate choice, not a claim of full cross-partition
transactional atomicity. See "Concurrency, locking, and crash safety"
below for what happens when a crash lands between writing a partition's
data, its `CURRENT` pointer, and the manifest.

## Concurrency, locking, and crash safety

`apply_incremental_update` acquires a per-`(symbol, timeframe)` advisory
file lock (`historical/locking.py::DatasetLock`, via
`dataset_lock_path(...)`) around the entire reconciliation, so two
processes updating the SAME dataset concurrently no longer race: the
second acquisition fails fast with `DatasetLockError` naming the current
holder's PID, hostname, and lock age, rather than silently corrupting or
last-writer-wins-overwriting the other's work. Two processes updating
DIFFERENT `(symbol, timeframe)` pairs never contend -- the lock path is
scoped per dataset, not global.

- **Acquisition is atomic**: the lock file is created with
  `O_CREAT | O_EXCL`, so exactly one of two racing processes wins.
- **Stale-lock recovery**: if a process crashes while holding the lock, the
  lock file is simply left behind (there is no way to distinguish "still
  running" from "crashed" from the filesystem alone). Any later acquirer
  that finds a lock file older than `DEFAULT_STALE_AFTER` (1 hour), or that
  cannot even parse the lock file's contents (corrupted mid-write), reclaims
  it -- loudly, via a `logger.warning` naming the reclaimed holder -- rather
  than deadlocking forever. This is a deliberate, documented tradeoff: a
  crashed process's lock does not permanently wedge the dataset, at the
  cost of a bounded (1 hour, by default) window where a genuinely still-
  running process's lock *could* be prematurely reclaimed by another
  process on the same host. For a single-writer-at-a-time operational
  model (one scheduled ingestion job per dataset), this window is not
  expected to be hit in practice; treat it as a knob (constructor parameter
  on `DatasetLock`) to tune if your operational cadence differs.
- **Crash between writing data, metadata, manifest, and success markers**:
  every content blob is written to a temp directory and renamed into place
  only after its own `_SUCCESS` marker is written and its checksum verified
  -- a partial write is never visible under its final path. `CURRENT` is
  flipped only after the blob is fully in place. The manifest is written
  last, after every year's `CURRENT` pointer has been updated. A crash at
  any point before the manifest write leaves the store in a valid state
  (some years possibly still pointing at their pre-update content, or with
  content written but `CURRENT` not yet flipped -- `list_years` requires
  `CURRENT` to exist, so such a year is simply excluded from casual listing
  until the update is re-run) and the manifest is never saved describing
  work that didn't durably complete. Re-running `apply_incremental_update`
  after such a crash is always safe (idempotent).
- **Readers never take the lock.** `DatasetLoader` reads are lock-free by
  design -- a long-running backtest reading a dataset does not block a
  concurrent ingestion update, and vice versa, because reads always resolve
  either through an atomically-flipped `CURRENT` pointer (casual reads) or
  through immutable, never-overwritten content blobs named explicitly in a
  manifest (pinned-version reads). There is no reader/writer conflict to
  synchronize because writers never mutate what a reader might already be
  looking at.

## Resampling semantics

`historical/resampling.py::resample_ohlcv` computes bucket boundaries by
pure UTC-epoch arithmetic (never `pandas.DataFrame.resample`, which has
well-known `label`/`closed` ambiguity). Standard aggregation: `open=first,
high=max, low=min, close=last, tick_volume=sum, real_volume=sum`; `spread`
is the MEAN across constituent bars (a point-in-time quoted cost, not a
conserved quantity -- averaging is the most representative single figure
for downstream cost modeling, a deliberate choice documented in the
module). A derived bar is `is_complete=True` only if its bucket's close
time is `<=` the source data's own coverage end -- mirroring
`TimeframeCursor`'s close-time reveal rule -- and `DerivedBarPolicy.
REJECT_INCOMPLETE` (the default) drops any trailing incomplete bucket
before it ever reaches canonical storage. Every derived bar also carries
`source_bar_count`, so a bucket that legitimately spans a session closure
(fewer contributing bars, but still "complete" because its time window
has fully elapsed) is visibly distinguishable from a genuinely incomplete
one.

## Configuring MT5

`config.historical_schemas.IngestionConfig` (extending, not replacing, the
existing `config.schemas` system) is the single validated object
`quant_platform.data_cli` needs. See `examples/ingestion_config.example.json`
for a complete, safe (no real credentials) example. Credentials
(`login`/`password`/`server`) are never stored in a config file --
`MT5SourceConfig.with_credentials_from_env()` fills them from
`MT5_LOGIN`/`MT5_PASSWORD`/`MT5_SERVER` environment variables at runtime,
and no logging statement anywhere in this pipeline includes a password
(enforced by a `caplog`-based regression test).

## Running without MT5

The `MetaTrader5` package is lazily imported only inside
`MT5HistoricalSource.connect()`, and only when no `client` was injected --
every other module in this pipeline, and the entire test suite, runs
without it installed. `historical.mt5_testing.FakeMt5Client` implements the
same `Mt5ClientProtocol` surface the real package presents (connection
lifecycle, `symbol_select`, `copy_rates_range`, error codes), with
deterministic, query-independent price generation (see
`generate_fake_rates`'s docstring for why "query-independent" specifically
matters for incremental-update testing). Inject it via
`MT5HistoricalSource(config, client=FakeMt5Client(...))` for any test,
notebook, or demo that needs realistic-shaped historical data without a
live terminal. Requesting real MT5 functionality without the package
installed and without an injected client raises `MissingDependencyError`
with an actionable message -- it never silently no-ops or fakes success.

## Real MT5 adapter verification procedure

Everything else in this document is verified against
`historical.mt5_testing.FakeMt5Client` -- a protocol-faithful fake, not a
real broker connection. Before trusting this pipeline against a real MT5
account, run the actual verification steps below against a live terminal.
None of this is optional or automatable away: a fake client cannot tell you
whether *your specific broker's* symbol aliasing or server timezone is
configured correctly in `IngestionConfig`.

**1. Smoke-test command.** Set `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER`
in the environment (never in the config file, per "Configuring MT5"
above), then run:

```bash
python -m quant_platform.data_cli smoke-test-mt5 --config config.json
```

This connects, requests one hour of the most recently closed bars for the
configured `source_symbol`/`timeframe`, disconnects, and prints diagnostics
-- it never fabricates a "success" if any step raised. Exit code is
non-zero on any failure (connection, symbol resolution, or fetch), with an
actionable stderr message.

**2. Diagnostics printed, and what to check:**

- *Connectivity*: "Connected." / "Disconnected." bracket the call; a
  connection failure surfaces the real MT5 error code and message from
  `last_error()` (e.g. invalid login, server unreachable, terminal not
  running) -- never swallowed.
- **Credentials are never printed or logged.** The smoke test only ever
  receives the already-constructed `HistoricalSource` (or, in
  `cmd_smoke_test_mt5`, builds one directly from env-sourced credentials
  that are never echoed) -- the same no-credential-logging guarantee
  enforced elsewhere in this pipeline applies here too (see the
  `caplog`-based regression test referenced above).
- *Symbol alias verification*: MT5 brokers commonly suffix or rename
  symbols (`XAUUSD` vs `XAUUSDm` vs `GOLD.spot`, etc.) -- `source_symbol`
  in your config must be the EXACT string your broker's terminal shows in
  Market Watch. The smoke test's fetch call passes `source_symbol`
  directly to `copy_rates_range`; if the alias is wrong, MT5 raises rather
  than silently returning empty data, and the smoke test surfaces that as
  a failure. A successful fetch is your confirmation the configured alias
  resolves correctly against this specific account.
- *Broker timezone verification*: the smoke test's requested window is
  computed directly from `pandas.Timestamp.now(tz="UTC")` floored to the
  timeframe grid, one full bar back -- i.e., "the most recently closed bar
  right now, in UTC." If your `server_timezone` (`FixedOffsetTimezone` /
  `NamedZoneTimezone`) is misconfigured, the returned bars' `open_time`
  (after conversion through `localize_broker_timestamps`) will look
  systematically hours off from "now" -- too far in the past (offset sign
  or magnitude wrong) or, if the server is ahead of UTC and the offset was
  applied backwards, implausibly in the future. Treat the printed most-
  recent bar timestamp as a manual check: it should look like "a few
  minutes to at most one bar-width ago," never off by a suspicious round
  number of hours.
- *Boundary/bar-count reconciliation*: the smoke test prints both the
  naive full-session expected bar count for the requested window (window
  duration / bar duration) and the actual row count returned, explicitly
  labeled so a mismatch can be interpreted correctly -- a lower actual
  count is **expected, not a bug**, if the window fell across a session
  close, weekend, or maintenance break (cross-check against your broker's
  published schedule and the `SessionCalendarConfig` you configured); an
  actual count *higher* than expected, or a large unexplained shortfall
  during a period you believe the market was open, is the signal worth
  investigating.

Because `MetaTrader5` is a Windows-only package requiring a running local
terminal, this exact command could not be executed against a live account
in this development environment -- the code path itself (argument
construction, connect/fetch/disconnect sequencing, diagnostic output, and
non-zero exit on failure) is exercised in
`tests/unit/test_data_cli.py::TestSmokeTestMt5` against `FakeMt5Client`,
including a real-dispatch-path test that confirms the actual CLI (not just
the fake) fails actionably when the `MetaTrader5` package is unavailable.
Running it against your real terminal is the verification step still owed
before production use, exactly as noted under "Known limitations" below.

## CLI commands

```bash
python -m quant_platform.data_cli ingest --config config.json --start 2024-01-01T00:00:00Z --end 2024-02-01T00:00:00Z
python -m quant_platform.data_cli validate --config config.json --symbol XAUUSD --timeframe M1 --start ... --end ...
python -m quant_platform.data_cli resample --config config.json --symbol XAUUSD --source-timeframe M1 --target-timeframe H1 --start ... --end ...
python -m quant_platform.data_cli inspect-manifest --config config.json --symbol XAUUSD --timeframe H1
python -m quant_platform.data_cli smoke-test-mt5 --config config.json
```

Every command exits non-zero on failure with an actionable stderr message
(never a raw traceback, never a credential); `ingest` and `resample` never
print a success message unless the requested work actually completed.

## Known limitations (honest, as measured)

- **Multi-partition updates are not cross-partition-atomic.** Each
  year-partition write is atomic; an update spanning several years is a
  sequence of independent atomic writes. Safe (nothing is ever corrupted)
  and recoverable (re-running is idempotent), not a single transaction.
- **The dataset lock is advisory and single-host.** `DatasetLock` prevents
  two `apply_incremental_update` calls against the same `(symbol,
  timeframe)` from racing, but it is a local lock file, not a
  distributed/networked lock -- it does not protect against two processes
  on two DIFFERENT machines pointed at the same store over a shared/
  network filesystem. It also has a bounded (1 hour, configurable)
  stale-lock reclamation window, meaning a genuinely still-running process
  that exceeds that window could theoretically have its lock reclaimed by
  another process on the same host; see "Concurrency, locking, and crash
  safety" above for the reasoning. `write_partition`/`write_snapshot`
  themselves remain individually atomic (temp dir + rename) regardless of
  locking, so no partition or snapshot can ever be left half-written by a
  race, lock or no lock.
- **Raw snapshot writes are not covered by the dataset lock.**
  `RawSnapshotStore.write_snapshot` is unconditionally safe under
  concurrency (each snapshot gets its own immutable, checksum-named
  directory, so a race just produces two distinct snapshots, never a
  corrupted one) and does not need coordination the way canonical-store
  `CURRENT`-pointer updates do.
- **The session/calendar model is intentionally small** (weekly window +
  daily breaks + ad-hoc holidays) -- adequate for spot/CFD XAUUSD, not a
  general exchange-calendar replacement.
- **Performance figures in `tests/performance/test_historical_pipeline_
  throughput.py` are single-machine, single-symbol measurements** (low-
  millions of rows); nothing here claims or has been measured at
  distributed/multi-symbol/billion-row scale.
- **Real MT5 connectivity has not been exercised against a live terminal
  in this environment** (none is available here) -- the adapter is
  implemented and fully tested against a protocol-faithful fake; treat
  live-terminal integration as environment-dependent verification still
  owed before production use against a real broker connection.
- **The MT5 adapter's per-call date-range ceiling is empirical, not
  documented by the vendor** -- `extraction_chunk_size_days` bounds it
  conservatively, but the true limit may vary by broker/account/package
  version.

## Recommended Milestone 3

1. Feature engineering directly on top of the canonical/derived datasets
   this milestone produces (the dataset loader + manifest system is
   designed to be the stable read interface for this).
2. Live/near-real-time ingestion (a "still-forming current bar" mode
   already exists via `DerivedBarPolicy.RETAIN_INCOMPLETE`; a genuinely
   live polling loop does not yet exist).
3. Multi-symbol batch ingestion/resampling orchestration (this milestone
   is single-symbol-at-a-time by design).
4. Running `smoke-test-mt5` (see "Real MT5 adapter verification
   procedure" above) against a real MT5 terminal connection in an
   environment that has one, to close the "environment-dependent" gap
   noted above -- the command and its diagnostics exist; only the live
   run against a real account is still owed.
5. If exchange-traded instruments are added, a proper exchange-calendar
   dependency in place of the current small session model.
