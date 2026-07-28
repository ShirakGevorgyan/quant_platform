"""Deterministic drawdown analysis (Milestone 5, Section 19) --
recomputed directly from a persisted `EquityCurve`, never inferred beyond
the available test period (a drawdown still underwater at the last
equity point is reported as `recovered=False`, never assumed to recover
later)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant_platform.backtesting.returns import EquityCurve
from quant_platform.core.exceptions import FinancialMetricError
from quant_platform.ml.persistence import as_json_list, parse_utc_timestamp, require_schema_version

DRAWDOWN_EPISODE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DrawdownEpisode:
    peak_timestamp: str
    trough_timestamp: str
    recovery_timestamp: str | None
    magnitude: float
    duration_bars: int
    recovered: bool

    def __post_init__(self) -> None:
        parse_utc_timestamp(self.peak_timestamp)
        parse_utc_timestamp(self.trough_timestamp)
        if self.recovery_timestamp is not None:
            parse_utc_timestamp(self.recovery_timestamp)
        if not math.isfinite(self.magnitude) or self.magnitude < 0.0:
            raise FinancialMetricError(f"DrawdownEpisode.magnitude must be finite and >= 0, got {self.magnitude!r}")
        if self.duration_bars < 0:
            raise FinancialMetricError(f"DrawdownEpisode.duration_bars must be >= 0, got {self.duration_bars}")
        if self.recovered != (self.recovery_timestamp is not None):
            raise FinancialMetricError("DrawdownEpisode.recovered must be True iff recovery_timestamp is set")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "peak_timestamp": self.peak_timestamp, "trough_timestamp": self.trough_timestamp,
            "recovery_timestamp": self.recovery_timestamp, "magnitude": self.magnitude,
            "duration_bars": self.duration_bars, "recovered": self.recovered,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> DrawdownEpisode:
        return cls(
            peak_timestamp=str(raw["peak_timestamp"]), trough_timestamp=str(raw["trough_timestamp"]),
            recovery_timestamp=(None if raw.get("recovery_timestamp") is None else str(raw["recovery_timestamp"])),
            magnitude=float(str(raw["magnitude"])), duration_bars=int(str(raw["duration_bars"])), recovered=bool(raw["recovered"]),
        )


@dataclass(frozen=True, slots=True)
class DrawdownReport:
    schema_version: int
    outer_fold_index: int
    equity_basis: str
    """`"gross"` or `"net"` -- which equity series this report was
    computed from (Section 8's "bar_return_sharpe"-style naming
    discipline applies here too: never one unlabeled "drawdown")."""
    episodes: tuple[DrawdownEpisode, ...]
    maximum_drawdown: float
    longest_drawdown_duration_bars: int
    current_ending_drawdown: float

    def __post_init__(self) -> None:
        if self.equity_basis not in ("gross", "net"):
            raise FinancialMetricError(f"DrawdownReport.equity_basis must be 'gross' or 'net', got {self.equity_basis!r}")
        for name, value in (("maximum_drawdown", self.maximum_drawdown), ("current_ending_drawdown", self.current_ending_drawdown)):
            if not math.isfinite(value) or value < 0.0:
                raise FinancialMetricError(f"DrawdownReport.{name} must be finite and >= 0, got {value!r}")
        if self.longest_drawdown_duration_bars < 0:
            raise FinancialMetricError(f"DrawdownReport.longest_drawdown_duration_bars must be >= 0, got {self.longest_drawdown_duration_bars}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index, "equity_basis": self.equity_basis,
            "episodes": [e.to_json_dict() for e in self.episodes], "maximum_drawdown": self.maximum_drawdown,
            "longest_drawdown_duration_bars": self.longest_drawdown_duration_bars, "current_ending_drawdown": self.current_ending_drawdown,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> DrawdownReport:
        require_schema_version(raw, supported=DRAWDOWN_EPISODE_SCHEMA_VERSION, context="DrawdownReport")
        return cls(
            schema_version=DRAWDOWN_EPISODE_SCHEMA_VERSION, outer_fold_index=int(str(raw["outer_fold_index"])),
            equity_basis=str(raw["equity_basis"]),
            episodes=tuple(DrawdownEpisode.from_json_dict(e) for e in as_json_list(raw.get("episodes") or [], field_name="episodes")),
            maximum_drawdown=float(str(raw["maximum_drawdown"])), longest_drawdown_duration_bars=int(str(raw["longest_drawdown_duration_bars"])),
            current_ending_drawdown=float(str(raw["current_ending_drawdown"])),
        )


def compute_drawdown_report(curve: EquityCurve, *, equity_basis: str, max_episodes: int = 10) -> DrawdownReport:
    """Section 19: a single deterministic pass over `curve.points`,
    tracking the running peak and detecting peak -> trough -> recovery
    episodes. `max_episodes` bounds how many of the LARGEST episodes are
    persisted in full (Section 19: "top bounded drawdown episodes") --
    `maximum_drawdown`/`longest_drawdown_duration_bars` are always computed
    over EVERY episode, never just the persisted subset."""
    if equity_basis not in ("gross", "net"):
        raise FinancialMetricError(f"compute_drawdown_report: equity_basis must be 'gross' or 'net', got {equity_basis!r}")
    values = [p.cumulative_gross_equity if equity_basis == "gross" else p.cumulative_net_equity for p in curve.points]
    timestamps = [p.timestamp for p in curve.points]

    episodes: list[DrawdownEpisode] = []
    peak_value = values[0]
    peak_ts = timestamps[0]
    peak_index = 0
    in_drawdown = False
    trough_value = peak_value
    trough_ts = peak_ts

    for i, (value, ts) in enumerate(zip(values, timestamps, strict=True)):
        if value >= peak_value:
            if in_drawdown:
                episodes.append(DrawdownEpisode(
                    peak_timestamp=peak_ts, trough_timestamp=trough_ts, recovery_timestamp=ts,
                    magnitude=(peak_value - trough_value) / peak_value, duration_bars=i - peak_index, recovered=True,
                ))
                in_drawdown = False
            peak_value, peak_ts, peak_index = value, ts, i
            trough_value, trough_ts = value, ts
        else:
            in_drawdown = True
            if value < trough_value:
                trough_value, trough_ts = value, ts

    if in_drawdown:
        episodes.append(DrawdownEpisode(
            peak_timestamp=peak_ts, trough_timestamp=trough_ts, recovery_timestamp=None,
            magnitude=(peak_value - trough_value) / peak_value, duration_bars=(len(values) - 1) - peak_index, recovered=False,
        ))

    maximum_drawdown = max((e.magnitude for e in episodes), default=0.0)
    longest_duration = max((e.duration_bars for e in episodes), default=0)
    current_ending_drawdown = (peak_value - values[-1]) / peak_value if values[-1] < peak_value else 0.0

    top_episodes = tuple(sorted(episodes, key=lambda e: e.magnitude, reverse=True)[:max_episodes])
    return DrawdownReport(
        schema_version=DRAWDOWN_EPISODE_SCHEMA_VERSION, outer_fold_index=curve.outer_fold_index, equity_basis=equity_basis,
        episodes=top_episodes, maximum_drawdown=maximum_drawdown, longest_drawdown_duration_bars=longest_duration,
        current_ending_drawdown=current_ending_drawdown,
    )


__all__ = ["DRAWDOWN_EPISODE_SCHEMA_VERSION", "DrawdownEpisode", "DrawdownReport", "compute_drawdown_report"]
