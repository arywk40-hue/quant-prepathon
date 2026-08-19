"""Deterministic, day-local Part 5 baseline backtest primitives."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.common.day_boundary import format_time_seconds, parse_time_seconds

from .schemas import DevelopmentScope


@dataclass(frozen=True)
class TransactionCostModel:
    """Fixed proportional execution-cost assumption.

    The supplied data contains no bid/ask, fee, or slippage fields.  The
    baseline therefore uses a parameterized 5 bps per-side execution cost.
    """

    entry_cost_bps: float = 5.0
    exit_cost_bps: float = 5.0
    fee_bps: float = 0.0

    @property
    def entry_rate(self) -> float:
        return (self.entry_cost_bps + self.fee_bps) / 10000.0

    @property
    def exit_rate(self) -> float:
        return (self.exit_cost_bps + self.fee_bps) / 10000.0

    def entry_cost(self, notional: float) -> float:
        return float(notional * self.entry_rate)

    def exit_cost(self, notional: float) -> float:
        return float(notional * self.exit_rate)

    def total_cost(self, notional: float) -> float:
        return self.entry_cost(notional) + self.exit_cost(notional)


@dataclass(frozen=True)
class StrategyConfig:
    target_horizon_seconds: int = 300
    holding_period_seconds: int = 300
    notional: float = 1.0
    starting_capital: float = 1.0
    max_exposure: float = 1.0
    signal_rule: str = "prediction > 0 LONG; prediction < 0 SHORT; prediction == 0 FLAT"
    position_rule: str = "one position at a time; signals before exact exit are ignored"
    entry_rule: str = "entry at the first raw price observation at or after signal timestamp"
    exit_rule: str = "exit at the exact raw timestamp entry_timestamp + 300 seconds within the same day"

    def validate(self) -> None:
        if self.target_horizon_seconds != 300 or self.holding_period_seconds != 300:
            raise ValueError("Part 5 baseline requires a 300-second target and holding period")
        if self.notional != 1.0 or self.starting_capital != 1.0 or self.max_exposure != 1.0:
            raise ValueError("Part 5 baseline requires fixed unit notional and exposure")


def validate_backtest_scope(scope: DevelopmentScope, validation_days: Iterable[int]) -> None:
    days = tuple(int(day) for day in validation_days)
    scope.assert_development_days(days)
    if set(days) & set(scope.missing_development_days):
        raise ValueError("backtest requests unavailable development days")
    if set(days) & set(scope.holdout_days):
        raise ValueError("backtest requests locked holdout days")


def signal_direction(prediction: float) -> int:
    if not np.isfinite(prediction):
        raise ValueError("non-finite prediction cannot generate a signal")
    return 1 if prediction > 0 else -1 if prediction < 0 else 0


def _price_arrays(price_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = {"Time", "Price"}
    if not required.issubset(price_frame.columns):
        raise ValueError("price frame must contain Time and Price")
    seconds = np.asarray([parse_time_seconds(value) for value in price_frame["Time"]], dtype=object)
    if any(value is None for value in seconds):
        raise ValueError("invalid price timestamp")
    timestamps = np.asarray(seconds, dtype=np.int64)
    prices = price_frame["Price"].to_numpy(dtype=float)
    if len(timestamps) == 0 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("price timestamps must be strictly increasing within a day")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("prices must be finite and positive")
    return timestamps, prices


def simulate_day(
    predictions: pd.DataFrame,
    price_frame: pd.DataFrame,
    *,
    window: str,
    day: int,
    strategy: StrategyConfig,
    costs: TransactionCostModel,
    direction_mode: str = "signal",
    random_generator: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate one day without interpolation or cross-day state."""

    strategy.validate()
    price_seconds, prices = _price_arrays(price_frame)
    required = {"day", "timestamp", "timestamp_seconds", "prediction"}
    if not required.issubset(predictions.columns):
        raise ValueError("predictions must contain day, timestamp, timestamp_seconds, and prediction")
    pred = predictions.copy().sort_values("timestamp_seconds", kind="mergesort")
    if set(pred["day"].astype(int).unique()) != {int(day)}:
        raise ValueError("prediction day does not match requested simulation day")
    if direction_mode not in {"signal", "random"}:
        raise ValueError("direction_mode must be signal or random")
    if direction_mode == "random" and random_generator is None:
        raise ValueError("random mode requires an explicit generator")

    rows: list[dict[str, object]] = []
    next_signal_seconds = -1
    trade_number = 0
    for item in pred.itertuples(index=False):
        signal_seconds = int(item.timestamp_seconds)
        if signal_seconds < next_signal_seconds:
            continue
        prediction = float(item.prediction)
        if direction_mode == "random":
            direction = int(random_generator.choice(np.asarray([-1, 1], dtype=np.int8)))  # type: ignore[union-attr]
        else:
            direction = signal_direction(prediction)
        if direction == 0:
            continue
        entry_idx = bisect_left(price_seconds.tolist(), signal_seconds)
        if entry_idx >= len(price_seconds):
            continue
        entry_seconds = int(price_seconds[entry_idx])
        exit_seconds = entry_seconds + strategy.holding_period_seconds
        exit_idx = bisect_left(price_seconds.tolist(), exit_seconds)
        if exit_idx >= len(price_seconds) or int(price_seconds[exit_idx]) != exit_seconds:
            # The absence of an exact same-day exit is an unavailable trade,
            # not a license to interpolate or carry state into another day.
            continue
        entry_price = float(prices[entry_idx])
        exit_price = float(prices[exit_idx])
        gross_return = float(direction * (exit_price / entry_price - 1.0))
        gross_pnl = gross_return * strategy.notional
        entry_cost = costs.entry_cost(strategy.notional)
        exit_cost = costs.exit_cost(strategy.notional)
        transaction_cost = entry_cost + exit_cost
        net_return = gross_return - transaction_cost / strategy.notional
        rows.append({
            "trade_id": f"{window}-D{day}-T{trade_number:05d}",
            "window": window,
            "day": int(day),
            "signal_timestamp": str(item.timestamp),
            "signal_timestamp_seconds": signal_seconds,
            "entry_timestamp": format_time_seconds(entry_seconds),
            "entry_timestamp_seconds": entry_seconds,
            "exit_timestamp": format_time_seconds(exit_seconds),
            "exit_timestamp_seconds": exit_seconds,
            "direction": direction,
            "side": "LONG" if direction > 0 else "SHORT",
            "position": float(direction * strategy.max_exposure),
            "notional": float(strategy.notional),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "prediction": prediction,
            "gross_return": gross_return,
            "gross_pnl": gross_pnl,
            "entry_cost": entry_cost,
            "exit_cost": exit_cost,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "net_pnl": gross_pnl - transaction_cost,
        })
        trade_number += 1
        next_signal_seconds = exit_seconds
    return pd.DataFrame(rows)


def simulate_passive_day(
    price_frame: pd.DataFrame,
    *,
    window: str,
    day: int,
    strategy: StrategyConfig,
    costs: TransactionCostModel,
) -> pd.DataFrame:
    strategy.validate()
    seconds, prices = _price_arrays(price_frame)
    if len(seconds) < 2:
        return pd.DataFrame()
    entry_seconds, exit_seconds = int(seconds[0]), int(seconds[-1])
    entry_cost = costs.entry_cost(strategy.notional)
    exit_cost = costs.exit_cost(strategy.notional)
    gross_return = float(prices[-1] / prices[0] - 1.0)
    transaction_cost = entry_cost + exit_cost
    return pd.DataFrame([{
        "trade_id": f"{window}-D{day}-PASSIVE",
        "window": window,
        "day": int(day),
        "signal_timestamp": format_time_seconds(entry_seconds),
        "signal_timestamp_seconds": entry_seconds,
        "entry_timestamp": format_time_seconds(entry_seconds),
        "entry_timestamp_seconds": entry_seconds,
        "exit_timestamp": format_time_seconds(exit_seconds),
        "exit_timestamp_seconds": exit_seconds,
        "direction": 1,
        "side": "LONG",
        "position": 1.0,
        "notional": float(strategy.notional),
        "entry_price": float(prices[0]),
        "exit_price": float(prices[-1]),
        "prediction": np.nan,
        "gross_return": gross_return,
        "gross_pnl": gross_return * strategy.notional,
        "entry_cost": entry_cost,
        "exit_cost": exit_cost,
        "transaction_cost": transaction_cost,
        "net_return": gross_return - transaction_cost / strategy.notional,
        "net_pnl": gross_return * strategy.notional - transaction_cost,
    }])


def daily_pnl_from_trades(
    trades: pd.DataFrame,
    *,
    days: Iterable[int],
    window: str,
    session_seconds: dict[int, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sequence, day in enumerate(days):
        day = int(day)
        subset = trades[trades["day"].astype(int) == day] if not trades.empty else trades
        active = int((subset["exit_timestamp_seconds"] - subset["entry_timestamp_seconds"]).sum()) if not subset.empty else 0
        denominator = int(session_seconds[day])
        rows.append({
            "window": window,
            "day": day,
            "gross_pnl": float(subset["gross_pnl"].sum()) if not subset.empty else 0.0,
            "transaction_cost": float(subset["transaction_cost"].sum()) if not subset.empty else 0.0,
            "net_pnl": float(subset["net_pnl"].sum()) if not subset.empty else 0.0,
            "trade_count": int(len(subset)),
            "turnover": float(2.0 * subset["notional"].sum()) if not subset.empty else 0.0,
            "active_exposure_seconds": active,
            "average_exposure": float(active / denominator) if denominator else 0.0,
            "sequence": sequence,
        })
    return pd.DataFrame(rows)


def _safe_sharpe(values: np.ndarray) -> float | None:
    if len(values) < 2 or float(np.std(values, ddof=1)) == 0.0:
        return None
    return float(np.mean(values) / np.std(values, ddof=1) * np.sqrt(252.0))


def _safe_sortino(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    downside_deviation = float(np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
    if downside_deviation == 0.0:
        return None
    return float(np.mean(values) / downside_deviation * np.sqrt(252.0))


def summarize_trades(
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    window: str,
    strategy: StrategyConfig,
) -> dict[str, object]:
    if trades.empty:
        returns = np.asarray([], dtype=float)
        gross = costs = net = 0.0
        win_rate = 0.0
        avg = median = std = 0.0
        turnover = 0.0
    else:
        returns = trades["net_return"].to_numpy(dtype=float)
        gross = float(trades["gross_pnl"].sum())
        costs = float(trades["transaction_cost"].sum())
        net = float(trades["net_pnl"].sum())
        win_rate = float((returns > 0).mean())
        avg = float(np.mean(returns))
        median = float(np.median(returns))
        std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        turnover = float(2.0 * trades["notional"].sum())
    daily_values = daily["net_pnl"].to_numpy(dtype=float) if not daily.empty else np.asarray([], dtype=float)
    curve = equity_curve_from_daily(daily, starting_capital=strategy.starting_capital)
    max_dd = float(curve["drawdown"].min()) if not curve.empty else 0.0
    return {
        "window": window,
        "strategy": "primary_sign_strategy",
        "trades": int(len(trades)),
        "gross_pnl": gross,
        "transaction_costs": costs,
        "net_pnl": net,
        "sharpe": _safe_sharpe(daily_values),
        "sortino": _safe_sortino(daily_values),
        "maximum_drawdown": max_dd,
        "turnover": turnover,
        "win_rate": win_rate,
        "average_trade_return": avg,
        "median_trade_return": median,
        "trade_return_std": std,
        "daily_pnl_std": float(np.std(daily_values, ddof=1)) if len(daily_values) > 1 else 0.0,
        "average_exposure": float(daily["average_exposure"].mean()) if not daily.empty else 0.0,
        "maximum_exposure": float(strategy.max_exposure) if not daily.empty else 0.0,
        "validation_days": int(len(daily)),
        "validation_rows": None,
    }


def equity_curve_from_daily(daily: pd.DataFrame, *, starting_capital: float = 1.0) -> pd.DataFrame:
    if daily.empty:
        return daily.copy()
    result = daily.copy()
    result["equity"] = starting_capital + result["net_pnl"].cumsum()
    result["running_max_equity"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["running_max_equity"] - 1.0
    return result
