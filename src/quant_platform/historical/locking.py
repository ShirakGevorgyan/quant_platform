"""Advisory dataset-level locking with stale-lock recovery.

`CanonicalStore`'s content-addressed design (see its module docstring)
already makes concurrent writes to the SAME `(symbol, timeframe)` safe from
corruption -- two writers submitting different content never collide (each
gets its own content-address), and the worst case of a `CURRENT`-pointer
race is "which write is casually current is nondeterministic", never data
loss. `DatasetLock` adds real mutual exclusion on top of that for the
normal case: this pipeline is designed for one writer at a time per
dataset, and a lock turns "two ingestion processes running against the
same dataset at once" from a silent race into an explicit, loud failure
(or an orderly wait), rather than leaving it to the storage layer's
"no corruption, but nondeterministic" fallback.

The lock is a plain file (`<dataset_dir>/.lock`) containing the holder's
PID, hostname, and UTC acquisition time. It is published atomically,
CONTENT INCLUDED: the full JSON payload is written to a private,
uniquely-named temp file first, then "published" at the shared lock path
via `os.link` (which fails atomically with `FileExistsError` if that path
already exists, on both POSIX and Windows) -- see `DatasetLock._try_
publish`'s own docstring for why a plain exclusive-CREATE-then-separate-
write is not enough: it lets a concurrent contender observe the lock file
mid-creation (created, but still empty) and misdiagnose a live lock as
corrupted. A lock older than `stale_after` is considered abandoned (the
holder crashed without releasing it) and can be forcibly reclaimed,
logging a warning; this is the "stale-lock recovery" half of the
contract. There is no distributed consensus here (this is a local-
filesystem advisory lock, appropriate for this platform's single-machine,
single-writer-at-a-time design target), and it does not protect against a
process that ignores it -- it protects cooperating callers
(`apply_incremental_update` always acquires it) from each other.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import pandas as pd

from quant_platform.core.exceptions import DatasetLockError
from quant_platform.core.json import canonical_json_bytes, parse_json_strict

logger = logging.getLogger(__name__)

_LOCK_FILE_NAME = ".lock"
DEFAULT_STALE_AFTER = pd.Timedelta(hours=1)


def dataset_lock_path(canonical_store_root: Path, *, symbol: str, timeframe_value: str) -> Path:
    """The standard lock file location for one `(symbol, timeframe)`
    dataset -- a sibling of that dataset's canonical directory, so it is
    visible/removable alongside the data it protects without digging
    through `CanonicalStore` internals."""
    from quant_platform.data.interfaces import DataSource

    safe_symbol = DataSource.sanitize_identifier(symbol, field_name="symbol")
    return canonical_store_root / "canonical" / f"symbol={safe_symbol}" / f"timeframe={timeframe_value}" / _LOCK_FILE_NAME


@dataclass(frozen=True, slots=True)
class LockInfo:
    pid: int
    hostname: str
    acquired_at: pd.Timestamp

    def to_json_dict(self) -> dict[str, object]:
        return {"pid": self.pid, "hostname": self.hostname, "acquired_at": self.acquired_at.isoformat()}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LockInfo:
        return cls(pid=int(str(raw["pid"])), hostname=str(raw["hostname"]), acquired_at=pd.Timestamp(str(raw["acquired_at"])))


class DatasetLock:
    """Context manager: `with DatasetLock(lock_path): ...` acquires on
    enter, releases on exit (including on exception). Acquisition fails
    fast with `DatasetLockError` if another (non-stale) holder has it --
    this pipeline chooses to fail loudly rather than block/retry, since a
    silently-long wait inside a batch ingestion job is worse than an
    immediate, actionable error.
    """

    def __init__(self, lock_path: Path, *, stale_after: pd.Timedelta = DEFAULT_STALE_AFTER) -> None:
        self._lock_path = lock_path
        self._stale_after = stale_after
        self._held = False

    def acquire(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        info = LockInfo(pid=os.getpid(), hostname=socket.gethostname(), acquired_at=pd.Timestamp.now(tz="UTC"))
        if self._try_publish(info):
            self._held = True
            logger.info("Dataset lock acquired: path=%s pid=%d", self._lock_path, info.pid)
            return
        self._handle_existing_lock(info)

    def _try_publish(self, info: LockInfo) -> bool:
        """Atomically publishes `info` as the lock file's FULL content at
        `self._lock_path`, or returns `False` if a lock file already
        exists there (never raises for that expected case).

        WHY NOT `os.open(path, O_CREAT | O_EXCL | O_WRONLY)` FOLLOWED BY
        A SEPARATE WRITE (the previous implementation)
        --------------------------------------------------------------
        That sequence is exclusive-CREATE atomic, but the file is
        observably EMPTY for the (short but real) window between the
        `open` succeeding and the subsequent `write` completing. A
        concurrent contender that lost the `open` race and falls into
        `_handle_existing_lock` can call `_read_existing_lock` during
        exactly that window, see an empty/unparseable file, and
        (correctly, given what it can observe) conclude the lock is
        corrupted -- reclaiming it by unlinking and recreating even
        though the original holder is still alive and about to write.
        If the original holder finishes writing and releases its own
        file descriptor before the "reclaimer" gets to its own unlink-
        and-recreate step, BOTH ends up believing they hold the lock
        (verified via a forced-interleaving reproduction; see this
        class's regression test for the exact sequence).

        Writing the FULL content to a private, uniquely-named temp file
        FIRST, then "publishing" it via `os.link` (which fails atomically
        with `FileExistsError` if the destination already exists, on
        both POSIX and Windows) means the shared lock path is NEVER
        observable in a content-less state: it either does not exist yet,
        or already has its complete, valid content the moment it exists
        at all."""
        tmp_path = self._lock_path.with_name(f".{self._lock_path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp_path.write_bytes(canonical_json_bytes(info.to_json_dict()))
        try:
            os.link(tmp_path, self._lock_path)
            return True
        except FileExistsError:
            return False
        finally:
            tmp_path.unlink(missing_ok=True)

    def _handle_existing_lock(self, new_info: LockInfo) -> None:
        existing = self._read_existing_lock()
        if existing is not None:
            age = pd.Timestamp.now(tz="UTC") - existing.acquired_at
            if age <= self._stale_after:
                raise DatasetLockError(
                    f"Dataset is locked by pid={existing.pid} on {existing.hostname} "
                    f"(acquired {existing.acquired_at}, {age} ago) -- refusing to proceed "
                    "concurrently. If that process is confirmed dead, wait for the lock to "
                    f"go stale (after {self._stale_after}) or remove {self._lock_path} manually.",
                    context={"lock_path": str(self._lock_path), "holder_pid": existing.pid, "age": str(age)},
                )
            logger.warning(
                "Reclaiming stale dataset lock: path=%s previous_holder_pid=%d age=%s (> stale_after=%s)",
                self._lock_path, existing.pid, age, self._stale_after,
            )
        else:
            logger.warning(
                "Reclaiming unreadable/corrupted dataset lock file: path=%s", self._lock_path
            )
        # Reclaim: remove the stale/corrupted lock and retry acquisition
        # once, via the SAME atomic write-then-link publish path `acquire`
        # itself uses. A second collision here means a genuine concurrent
        # acquisition race lost -- surface that rather than looping.
        #
        # `missing_ok=True` alone is not enough: it only swallows
        # `FileNotFoundError` (this path was already gone). Found via a
        # deterministic two-thread reclaim-race regression test: on
        # Windows, when a SECOND thread/process concurrently unlinks the
        # SAME path a moment after the first, NTFS's delete semantics can
        # raise `PermissionError` (access denied / sharing violation)
        # instead of "not found" for the loser of that micro-race. That
        # is not a real failure to guard against here -- the actual
        # arbiter of "who gets the lock" is `_try_publish`'s `os.link`
        # call immediately below, which is atomic regardless of whether
        # THIS unlink succeeded, failed benignly, or was a no-op. Any
        # `OSError` here is therefore treated exactly like "already
        # gone": proceed to `_try_publish` and let ITS atomicity decide.
        with contextlib.suppress(OSError):
            self._lock_path.unlink(missing_ok=True)
        if not self._try_publish(new_info):
            raise DatasetLockError(
                f"Lost a race reclaiming stale lock {self._lock_path} against another process",
                context={"lock_path": str(self._lock_path)},
            )
        self._held = True
        logger.info("Dataset lock reclaimed and acquired: path=%s pid=%d", self._lock_path, new_info.pid)

    def _read_existing_lock(self) -> LockInfo | None:
        """`None` means "treat this lock as reclaimable" -- deliberately
        the SAME outcome for every failure mode below, a decision made
        (not merely inherited) when this module was migrated onto
        `core.json.parse_json_strict`/`canonical_json_bytes`. This is
        fail-OPEN-to-reclaim by design, safe specifically BECAUSE
        `_try_publish`'s write-temp-then-`os.link` protocol (unchanged by
        this migration) makes a lock file observable in exactly two
        states: fully absent, or fully present with complete, valid
        content -- there is no code path in THIS module that could ever
        leave a live, currently-held lock in any of the states below.
        Reaching one of them therefore means either (a) a genuine benign
        race (the lock was concurrently released/reclaimed between the
        existence check and this read -- os.link's own atomicity is what
        actually adjudicates any real collision, not this method), or
        (b) external interference (hand-editing, disk corruption) this
        local-filesystem advisory lock never claimed to defend against
        (see module docstring: "does not protect against a process that
        ignores it"). Explicitly decided per failure mode:

          * malformed lock JSON (`parse_json_strict` -> `ValueError`):
            reclaimable.
          * non-object root, e.g. a JSON array (`LockInfo.from_json_dict`
            indexing a non-dict -> `TypeError`): reclaimable. Previously
            UNCAUGHT here (a real, now-fixed gap: the pre-migration
            except tuple had no `TypeError`, so this specific case would
            have raised out of `_read_existing_lock` uncaught instead of
            being treated as reclaimable).
          * NaN/Infinity fields (`parse_json_strict` -> `ValueError`,
            rejected at parse time -- previously would have reached
            `int(str(raw["pid"]))`'s own `ValueError` instead, same
            outcome either way): reclaimable.
          * duplicate keys (`parse_json_strict` -> `ValueError`,
            rejected at parse time -- previously silently resolved to
            the last occurrence and NOT treated as corruption; the
            platform-wide duplicate-key-rejection policy applies here
            too): reclaimable.
          * missing owner fields, e.g. no `pid` (`KeyError`):
            reclaimable.
          * invalid/unparseable timestamp (`pd.Timestamp(...)` ->
            `ValueError`): reclaimable.
          * partial/truncated file (malformed JSON, same as the first
            case -- and per the atomicity argument above, only reachable
            via external interference, never this module's own write
            path): reclaimable.
          * the file vanished between the caller's existence check and
            this read (`OSError`/`FileNotFoundError`): reclaimable --
            there is genuinely nothing there any more.

        None of these silently treats a lock as reclaimable WITHOUT
        first attempting `os.link`'s own atomic collision check in
        `_try_publish` -- a "reclaim" that loses that race still raises
        `DatasetLockError` (see `_handle_existing_lock`), so a live
        holder that races past this exact window is still protected."""
        try:
            raw = parse_json_strict(self._lock_path.read_text(encoding="utf-8"))
            return LockInfo.from_json_dict(raw)
        except (OSError, UnicodeDecodeError, KeyError, ValueError, TypeError):
            return None

    def release(self) -> None:
        if self._held:
            self._lock_path.unlink(missing_ok=True)
            self._held = False
            logger.info("Dataset lock released: path=%s", self._lock_path)

    def __enter__(self) -> DatasetLock:
        self.acquire()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        self.release()


__all__ = ["DEFAULT_STALE_AFTER", "DatasetLock", "LockInfo"]
