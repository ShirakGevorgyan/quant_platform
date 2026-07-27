# Durable-Persistence and Verification Security (Milestone 4D.1)

Cross-package hardening of every durable JSON read/write path in the
platform. Two parts: an initial pass hardening `execution`, `ml`
(including `ml/model_zoo`), and `optimization`; a completion pass
extracting the shared primitives into a dependency-neutral home
(`quant_platform.core.json`) so `historical` and `features` -- which
cannot safely depend on `quant_platform.ml` -- could be migrated onto the
same authoritative implementation too, closing the one gap the first pass
left open.

## 1. Why the persistence primitives live in `core`

`quant_platform.ml` already depends on `quant_platform.historical` and
`quant_platform.features` (`ml.experiment_manager`/`ml.validation` import
`features.manifests`; `ml.concurrency` imports `historical.locking`).
`historical`/`features` must never depend back on `ml` -- that would be a
real circular dependency, not a hypothetical one: a direct import-chain
check confirms `features.manifests` importing `quant_platform.ml.
persistence` would trigger `quant_platform/ml/__init__.py`'s own import
chain (`ml.experiment_manager`/`ml.validation` importing `features.
manifests` again, mid-import) and fail.

`quant_platform.core` was already the platform's dependency-free
foundation (`core/__init__.py`'s own docstring: "Nothing in this
subpackage performs I/O or depends on any other `quant_platform`
subpackage"), and already houses every domain's exceptions
(`core.exceptions`). Moving the durable-JSON primitives there -- a new
`quant_platform/core/json.py` module -- gives `historical`, `features`,
`ml`, `execution`, and `optimization` a common ancestor to depend
DOWNWARD on, instead of sideways or upward on each other's higher layers.

## 2. Dependency direction (verified, not just documented)

```
                     core.json  (stdlib only)
                         ^
              ----------------------------
              |                          |
         historical, features        ml (re-exports core.json
                 ^                    via ml.persistence)
                 |                        ^
                 -------------------------|
                              |
                    execution, optimization
```

`tests/unit/test_architecture_boundaries.py` proves this two ways:
statically (AST-parsing `core/json.py`'s own `import`/`from` statements:
zero `quant_platform.*` references at all, not even `quant_platform.
core`'s own siblings) and dynamically (importing every top-level package,
in multiple orders including `historical`/`features` before `ml` even
exists in `sys.modules`, in fresh subprocesses -- a cycle that only
manifests for one particular import order would not be caught statically
alone).

## 3. The authoritative parser and writer

`quant_platform.core.json` is the **single, exclusive** implementation of:

- `canonical_json_bytes(payload) -> bytes` -- sorted keys, compact
  separators, `ensure_ascii=True`, `allow_nan=False`.
- `parse_json_strict(text) -> Any` -- rejects the bare `NaN`/`Infinity`/
  `-Infinity` tokens Python's `json` module otherwise accepts as a
  non-standard extension, AND rejects duplicate object keys (Milestone
  4D.1 completion; see Section 6).
- `sha256_hex_bytes(data) -> str`.
- `write_json_atomic(path, payload) -> None` -- temp-file-then-rename;
  raises only `ValueError`/`OSError`, never a domain-specific exception
  (translating a failure into one is every caller's job -- `core.json`
  has no opinion on which subpackage or artifact category is involved).

No second implementation of any of these four functions exists anywhere
in this codebase.

## 4. Compatibility re-exports through `ml.persistence`

`quant_platform.ml.persistence.{canonical_json_bytes, parse_json_strict,
sha256_hex_bytes, write_json_atomic}` are literal re-exports:
`ml.persistence.parse_json_strict is core.json.parse_json_strict` holds
by construction (`tests/unit/test_architecture_boundaries.py::
TestMlPersistenceDelegatesToCoreJson` asserts object identity for all
four, and separately asserts `ml/persistence.py`'s own source does not
redefine either function name). Every pre-existing `from quant_platform.
ml.persistence import parse_json_strict` (or `canonical_json_bytes`)
call site across `execution`/`ml`/`optimization` continues to work
completely unchanged.

`ml.persistence` itself still owns everything specific to the ML layer
that `core.json` deliberately has no opinion on: `assert_within_root`
(path-containment), `require_schema_version`, `as_json_dict`/
`as_json_list` (JSON-primitive type narrowing), UTC timestamp formatting,
and `read_json_file` -- a convenience wrapper built ON TOP of `core.json.
parse_json_strict` that additionally translates a missing/unreadable/
malformed file into `ArtifactCorruptionError`.

## 5. `historical`/`features` migration

Every durable JSON read in `historical/{manifest,canonical_store,
raw_store,locking}.py` and `features/manifests.py` now calls `core.json.
parse_json_strict` (previously plain `json.loads`). `historical/
locking.py`'s `_read_existing_lock` gets its own detailed treatment below
(Section 7). Domain-specific exceptions (`ManifestError`, `SnapshotError`,
`ResearchDatasetError`, `DatasetLockError`) are preserved and, where a
gap was found, widened: `features.manifests.ResearchDatasetStore.{read_
artifacts,read_preprocessing}` previously had **no** exception handling
at all around their reads (a corrupted file raised a raw, untranslated
exception); `historical`'s three manifest/metadata loaders previously
caught only `json.JSONDecodeError`, not `UnicodeDecodeError` or a
`from_json_dict` structural failure (`KeyError`/`TypeError` on a
validly-parsed-but-wrong-shape payload). All now catch the complete set
and raise the correct domain exception. No raw `JSONDecodeError`,
`UnicodeDecodeError`, `ValueError`, `TypeError`, or `KeyError` escapes any
public loader in either package.

**Identity-critical writes are the one deliberate exception to "always
use `canonical_json_bytes`."** Six functions across `features/{manifests,
dataset_builder,models,normalization,registry}.py` compute a permanent,
widely-propagated identity (`content_id`, `dataset_id`,
`feature_registry_fingerprint`, a `FeatureSpec`/pipeline fingerprint) by
hashing a `json.dumps(...)`-serialized payload directly. Migrating these
to `canonical_json_bytes`'s compact-separator form would change every one
of those identities for newly-computed (but logically identical) input --
exactly the regression Section 8 exists to prevent. Each instead gets
**only** `allow_nan=False` added, with every other parameter (indent,
separators, `default=str` where present) left completely untouched, so
bytes for any payload that was already finite are provably unchanged (see
Section 8) while NaN/Infinity is still rejected. Each such function's own
docstring explains this explicitly, cross-referencing this document.

`features.manifests.write_artifacts`'s `metadata.json` write (which does
**not** feed any content-hash) and `ResearchManifestStore`'s manifest
file (whose own bytes are likewise never re-hashed for identity) were
both safely migrated to full `canonical_json_bytes`.

## 6. Duplicate-key policy: **reject** (not last-value-wins)

Chosen, not merely retained: `core.json.parse_json_strict` installs an
`object_pairs_hook` that raises `ValueError` the moment any JSON object
(at any nesting depth) contains a repeated key, rather than Python's
`json` module's default "last occurrence silently wins."

**Provably safe to introduce.** `canonical_json_bytes` serializes from a
Python `dict`, which cannot structurally hold a duplicate key -- so
`canonical_json_bytes`'s own output can never contain one. Every
legitimately platform-written artifact was therefore already
duplicate-key-free before this policy existed; rejecting a duplicate key
can only ever reject a file that was hand-edited or corrupted outside
this platform's own write path. `tests/unit/ml/test_persistence.py::
TestParseJsonStrict::{test_duplicate_keys_are_rejected,
test_duplicate_keys_rejected_at_any_nesting_depth,
test_non_duplicate_keys_are_unaffected}` and the corruption-test suites
in `tests/unit/historical/` and `tests/unit/features/` cover this for
every migrated store.

## 7. Lock-file parsing semantics (`historical.locking`)

`DatasetLock._read_existing_lock` returns `None` (meaning "treat this
lock as reclaimable") for every one of the following, an explicit
per-case decision, not an accident of exception-tuple width:

| Case | Mechanism | Outcome |
|---|---|---|
| Malformed JSON | `parse_json_strict` -> `ValueError` | reclaimable |
| Non-object root (e.g. a JSON array) | `LockInfo.from_json_dict` indexing a non-dict -> `TypeError` | reclaimable (previously **uncaught** -- the pre-migration except tuple had no `TypeError`; a real, now-fixed gap) |
| NaN/Infinity fields | `parse_json_strict` rejects the token at parse time (previously would have reached `int(str(raw["pid"]))`'s own `ValueError` instead -- same outcome, different rejection point) | reclaimable |
| Duplicate keys | `parse_json_strict` -> `ValueError` | reclaimable |
| Missing owner field (e.g. no `pid`) | `KeyError` | reclaimable |
| Invalid/unparseable timestamp | `pd.Timestamp(...)` -> `ValueError` | reclaimable |
| Partial/truncated file | malformed JSON, same as row 1 -- only reachable via external interference; `_try_publish`'s write-temp-then-`os.link` protocol (unchanged) never leaves a live lock partially visible | reclaimable |
| Path vanished between existence check and read | `OSError` | reclaimable |

This is fail-**open**-to-reclaim by design, safe specifically because the
already-audited atomic `os.link` publication protocol (untouched by this
migration) is what actually adjudicates any real collision -- a
"reclaim" that loses the follow-up `os.link` race still raises
`DatasetLockError`, so a genuinely live holder is never silently
double-acquired against.

**A genuine, previously-latent bug was found and fixed while writing
this migration's concurrency regression tests** (not introduced by the
parser swap): `_handle_existing_lock`'s `self._lock_path.unlink(missing_
ok=True)` -- the "clear the way before reclaiming" step -- can, on
Windows, raise `PermissionError` (access denied / sharing violation)
instead of `FileNotFoundError` when two threads/processes race to delete
the same path within the same instant; `missing_ok=True` only swallows
`FileNotFoundError`. A deterministic, barrier-synchronized two-thread
test forcing exactly this race (`tests/unit/historical/test_locking.py::
TestNoDoubleAcquisitionUnderForcedInterleaving::
test_exactly_one_winner_when_two_threads_race_to_reclaim_a_corrupted_lock`)
reproduced it directly (stress-tested clean across 60 repeated runs after
the fix). Fixed by wrapping the unlink in `contextlib.suppress(OSError)`:
the unlink is best-effort only -- `_try_publish`'s subsequent `os.link`
call is the real, atomic arbiter of the race regardless of whether this
unlink succeeded, failed benignly, or was a no-op.

## 8. Canonical-byte and identity compatibility

Proven two independent ways in `tests/unit/test_persistence_identity_
compatibility.py`:

1. **Pre-existing pinned golden hash.** `tests/unit/ml/test_experiment_
   identity.py`'s `GOLDEN_EXPERIMENT_ID` was fixed in this repository
   before this migration existed and still passes unmodified.
2. **Live "new vs. reconstructed-old" comparison**, for every fingerprint
   function actually touched (Section 5's six identity-critical writes,
   `core.json.canonical_json_bytes` itself against a hardcoded byte-string
   golden vector, `ExperimentIdentity`, and `OptimizationIdentity`): each
   test recomputes what the pre-migration code would have produced,
   inline, and asserts byte-for-byte/hash-for-hash equality with the
   current code's output.

No schema version was bumped anywhere; no serialization format changed
for any payload that was already legitimately writable.

## 9. Error-translation contract (unchanged from the first pass)

No public API in `execution`, `ml`, `optimization`, `historical`, or
`features` lets a raw `json.JSONDecodeError`, bare `ValueError`,
`UnicodeDecodeError`, `KeyError`, or `TypeError` escape a durable-artifact
read. See each package's own modules for their local "everything a
corrupted artifact can legitimately raise" tuple and its translation to
the narrowest correct domain exception.

## 10. Trusted vs. untrusted artifact boundary (unchanged from the first pass)

Content-addressed storage is content-verified (SHA-256 re-hashed on every
read) but not semantically trusted -- a new, internally self-consistent
payload written under its own correctly-computed hash is a materially
different threat than bit rot, and is what schema/category/identity
verification exists to catch. Manifests are plain, mutable,
non-content-addressed files, trusted only as far as they re-verify
against the artifacts they reference.

## 11. Unsupported/forbidden serialization formats (unchanged, re-verified)

No `pickle`, `joblib.dump`/`load`, `cloudpickle`, `dill`, `yaml.load`/
`yaml.unsafe_load`, bare `eval`, or `exec` appears anywhere in
`src/quant_platform/` (re-verified by repository-wide scan as part of
this completion pass -- see the delivery report's scan table).

## Limitations (honest, current)

- **Model-parameter-level finiteness** (coefficients, intercepts,
  baseline constants inside `ml.model_zoo` fitted-model envelopes) is
  still not validated -- only envelope parsing/schema safety (already
  fully compliant: every `model_zoo` serializer already used
  `parse_json_strict`/`canonical_json_bytes`). Judged out of scope: model
  -content validation, not persistence/JSON hardening.
- **Optuna sampler/study state** (`optimization.study`) round-trips
  through Optuna's own internal mechanisms, not `core.json` -- unchanged,
  out of scope, covered separately by the version-compatibility binding
  described in `docs/optimization_engine.md`.
- **CLI JSON output (`--format json`) is fail-closed, not fail-soft**, on
  a non-finite derived field: `allow_nan=False` (added to all three
  `ml_cli.py` call sites in the first pass) means such a report now
  raises rather than ever printing invalid JSON. No code path has been
  observed to actually produce such a value; the fix is preventative.
- **`historical`/`features`' NaN-rejection gap is now closed.** (Resolved
  by this completion pass -- previously the last open limitation in this
  document.)
