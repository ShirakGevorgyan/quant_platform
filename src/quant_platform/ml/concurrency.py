"""Thin ML-boundary adapter over `historical.locking.DatasetLock`.

Reuses `DatasetLock` directly, for the exact reason `manifests.py`/
`tracking.py`'s own module docstrings already give (do not duplicate
locking logic) -- this module adds nothing to the locking mechanism
itself, no new lock file format, no new reclaim policy. Its ONLY job is
translating what an ML caller sees when lock ACQUISITION fails, so the ML
public boundary never leaks:

  - a raw `DatasetLockError` -- a `historical`-layer (Milestone 2)
    exception type that an ML caller should not need to know about or
    catch by name, or
  - a raw `PermissionError`/`OSError` -- the documented, rare Windows
    race in `DatasetLock._handle_existing_lock`'s stale-lock-reclaim path
    under many-way simultaneous FIRST lock acquisition (see that class's
    docstring, and `docs/ml_infrastructure.md`'s "Concurrency" section).

into a single, ML-native `ExperimentLockError`, with the original
exception preserved as `__cause__` (never discarded).

WHAT THIS DOES NOT TOUCH
--------------------------------------------------------------------------
Only the ACQUIRE step is wrapped. Failures raised by the PROTECTED BODY
(the code running inside the `with experiment_lock(...):` block -- e.g. a
genuine disk-full error while writing a manifest) propagate completely
untouched, with their original type intact: this module does not swallow,
translate, or otherwise touch a genuine filesystem failure that has
nothing to do with lock acquisition.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from quant_platform.core.exceptions import DatasetLockError, ExperimentLockError
from quant_platform.historical.locking import DatasetLock


@contextmanager
def experiment_lock(lock_path: Path) -> Iterator[None]:
    """`with experiment_lock(path): ...` -- acquires `DatasetLock(path)`,
    translating a contested/unreadable/racing lock into `ExperimentLockError`
    (see module docstring), then runs the protected body, then always
    releases on the way out (including on exception from the body)."""
    lock = DatasetLock(lock_path)
    try:
        lock.acquire()
    except DatasetLockError as exc:
        raise ExperimentLockError(
            f"Could not acquire experiment lock at {lock_path}: {exc}",
            context={"lock_path": str(lock_path)},
        ) from exc
    except OSError as exc:
        raise ExperimentLockError(
            f"Experiment lock acquisition at {lock_path} hit an unexpected filesystem race "
            "(see historical.locking.DatasetLock's documented Windows stale-lock-reclaim "
            f"limitation): {exc}",
            context={"lock_path": str(lock_path)},
        ) from exc
    try:
        yield
    finally:
        lock.release()


__all__ = ["experiment_lock"]
