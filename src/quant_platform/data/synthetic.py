"""Deterministic synthetic OHLCV generation for tests, examples, and demos
that must not depend on a real market-data connection.

Uses a seeded geometric Brownian motion path sampled at `sub_steps`
intra-bar resolution so the resulting high/low genuinely bound the open/
close (rather than being derived after the fact with an ad-hoc noise term
that could violate OHLC invariants).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from quant_platform.core.time_utils import to_pandas_freq
from quant_platform.core.types import Timeframe

_SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


@dataclass(frozen=True, slots=True)
class SyntheticDataConfig:
    start: datetime
    periods: int
    timeframe: Timeframe
    initial_price: float = 100.0
    annualized_drift: float = 0.0
    annualized_volatility: float = 0.20
    seed: int = 42
    sub_steps: int = 4
    base_volume: float = 1_000.0
    volume_volatility: float = 0.30

    def __post_init__(self) -> None:
        if self.periods <= 0:
            raise ValueError(f"periods must be positive, got {self.periods}")
        if self.initial_price <= 0:
            raise ValueError(f"initial_price must be positive, got {self.initial_price}")
        if self.annualized_volatility < 0:
            raise ValueError(
                f"annualized_volatility must be non-negative, got {self.annualized_volatility}"
            )
        if self.sub_steps < 1:
            raise ValueError(f"sub_steps must be >= 1, got {self.sub_steps}")
        if self.base_volume <= 0:
            raise ValueError(f"base_volume must be positive, got {self.base_volume}")


def generate_ohlcv(config: SyntheticDataConfig) -> pd.DataFrame:
    """Generate a canonical-schema synthetic OHLCV DataFrame. Fully
    deterministic for a given `config.seed` -- calling this twice with the
    same config always produces bit-identical output."""
    rng = np.random.default_rng(config.seed)

    n = config.periods
    sub_steps = config.sub_steps
    dt_years = config.timeframe.duration.total_seconds() / _SECONDS_PER_YEAR
    sub_dt_years = dt_years / sub_steps

    mu = config.annualized_drift
    sigma = config.annualized_volatility

    total_path_points = n * sub_steps + 1
    innovations = rng.standard_normal(total_path_points - 1)
    log_returns = (mu - 0.5 * sigma**2) * sub_dt_years + sigma * np.sqrt(sub_dt_years) * innovations
    log_price_path = np.concatenate(([0.0], np.cumsum(log_returns)))
    price_path = config.initial_price * np.exp(log_price_path)

    # Every bar's OHLC is derived from its own (sub_steps + 1)-point window
    # of the shared path, so high/low provably bound open/close by construction.
    windows = np.lib.stride_tricks.sliding_window_view(price_path, sub_steps + 1)[::sub_steps]

    opens = windows[:, 0]
    closes = windows[:, -1]
    highs = windows.max(axis=1)
    lows = windows.min(axis=1)

    log_volume_noise = rng.normal(loc=0.0, scale=config.volume_volatility, size=n)
    volumes = config.base_volume * np.exp(log_volume_noise)

    open_times = pd.date_range(
        start=config.start, periods=n, freq=to_pandas_freq(config.timeframe), tz="UTC"
    )

    return pd.DataFrame(
        {
            "open_time": open_times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
