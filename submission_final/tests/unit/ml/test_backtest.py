import numpy as np
import pandas as pd
import pytest

from src.ebx.ml.backtest import (
    StrategyConfig,
    TransactionCostModel,
    daily_pnl_from_trades,
    signal_direction,
    simulate_day,
    summarize_trades,
    validate_backtest_scope,
)
from src.ebx.ml.schemas import DevelopmentScope


def _scope():
    return DevelopmentScope(85, tuple(range(1, 65)) + tuple(range(80, 86)), tuple(range(65, 80)), tuple(range(86, 109)))


def _prices(day=45, n=1000):
    return pd.DataFrame({"Time": [f"00:00:{i:02d}" if i < 60 else f"00:{i//60:02d}:{i%60:02d}" for i in range(n)], "Price": np.linspace(100, 101, n)})


def test_signal_direction_and_exact_holding_period():
    prices = pd.DataFrame({"Time": ["00:00:00", "00:05:00", "00:10:00"], "Price": [100.0, 101.0, 100.0]})
    pred = pd.DataFrame({"day": [45, 45, 45], "timestamp": ["00:00:00"] * 3, "timestamp_seconds": [0, 1, 300], "prediction": [1.0, -1.0, -1.0]})
    trades = simulate_day(pred, prices, window="W1", day=45, strategy=StrategyConfig(), costs=TransactionCostModel())
    assert signal_direction(1.0) == 1 and signal_direction(-1.0) == -1 and signal_direction(0.0) == 0
    assert len(trades) == 2
    assert (trades["exit_timestamp_seconds"] - trades["entry_timestamp_seconds"]).tolist() == [300, 300]


def test_cost_and_pnl_accounting():
    prices = pd.DataFrame({"Time": ["00:00:00", "00:05:00"], "Price": [100.0, 101.0]})
    pred = pd.DataFrame({"day": [45], "timestamp": ["00:00:00"], "timestamp_seconds": [0], "prediction": [1.0]})
    trade = simulate_day(pred, prices, window="W1", day=45, strategy=StrategyConfig(), costs=TransactionCostModel()).iloc[0]
    assert trade.gross_pnl == pytest.approx(0.01)
    assert trade.transaction_cost == pytest.approx(0.001)
    assert trade.net_pnl == pytest.approx(0.009)


def test_no_exact_exit_does_not_cross_day_or_interpolate():
    prices = pd.DataFrame({"Time": ["23:59:58", "23:59:59"], "Price": [100.0, 101.0]})
    pred = pd.DataFrame({"day": [45], "timestamp": ["23:59:58"], "timestamp_seconds": [86398], "prediction": [1.0]})
    assert simulate_day(pred, prices, window="W1", day=45, strategy=StrategyConfig(), costs=TransactionCostModel()).empty


def test_daily_aggregation_and_scope_protection():
    with pytest.raises(ValueError, match="non-development"):
        validate_backtest_scope(_scope(), [86])
    with pytest.raises(ValueError, match="non-development"):
        validate_backtest_scope(_scope(), [65])
    trades = pd.DataFrame([{"day": 45, "gross_pnl": .01, "transaction_cost": .001, "net_pnl": .009, "notional": 1.0, "entry_timestamp_seconds": 0, "exit_timestamp_seconds": 300, "net_return": .009}])
    daily = daily_pnl_from_trades(trades, days=[45, 46], window="W1", session_seconds={45: 1000, 46: 1000})
    assert daily.trade_count.tolist() == [1, 0]
    assert daily.net_pnl.tolist() == pytest.approx([.009, 0.0])


def test_target_column_cannot_change_simulation():
    prices = pd.DataFrame({"Time": ["00:00:00", "00:05:00"], "Price": [100.0, 101.0]})
    base = pd.DataFrame({"day": [45], "timestamp": ["00:00:00"], "timestamp_seconds": [0], "prediction": [1.0], "target": [-999.0]})
    changed = base.assign(target=999.0)
    a = simulate_day(base, prices, window="W1", day=45, strategy=StrategyConfig(), costs=TransactionCostModel())
    b = simulate_day(changed, prices, window="W1", day=45, strategy=StrategyConfig(), costs=TransactionCostModel())
    pd.testing.assert_frame_equal(a, b)


def test_random_direction_is_deterministic_and_strategy_is_fixed():
    prices = pd.DataFrame({"Time": ["00:00:00", "00:05:00"], "Price": [100.0, 101.0]})
    pred = pd.DataFrame({"day": [45], "timestamp": ["00:00:00"], "timestamp_seconds": [0], "prediction": [1.0]})
    a = simulate_day(pred, prices, window="W1", day=45, strategy=StrategyConfig(), costs=TransactionCostModel(), direction_mode="random", random_generator=np.random.default_rng(10))
    b = simulate_day(pred, prices, window="W1", day=45, strategy=StrategyConfig(), costs=TransactionCostModel(), direction_mode="random", random_generator=np.random.default_rng(10))
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError, match="300-second"):
        StrategyConfig(holding_period_seconds=60).validate()
