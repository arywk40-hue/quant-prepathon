"""ML Phase 4: one pre-specified development-only baseline backtest."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.coverage import load_price_day  # noqa: E402
from src.ebx.ml.backtest import (  # noqa: E402
    StrategyConfig,
    TransactionCostModel,
    daily_pnl_from_trades,
    equity_curve_from_daily,
    simulate_day,
    simulate_passive_day,
    summarize_trades,
    validate_backtest_scope,
)
from src.ebx.ml.cache import sha256_file, write_json  # noqa: E402
from src.ebx.ml.schemas import audited_scope  # noqa: E402
from src.ebx.ml.temporal_robustness import TEMPORAL_WINDOWS, validate_temporal_windows  # noqa: E402


SEED = 20260818


def _write_figures(figures_root: Path, daily: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame) -> None:
    destination = figures_root / "ml_phase4"
    destination.mkdir(parents=True, exist_ok=True)
    if not equity.empty:
        plt.figure(figsize=(9, 4))
        plt.plot(np.arange(len(equity)), equity["equity"], marker="o")
        plt.xlabel("Validation-day sequence")
        plt.ylabel("Normalized equity")
        plt.title("Part 5 baseline cumulative equity (development)")
        plt.tight_layout()
        plt.savefig(destination / "equity_curve.png", dpi=150)
        plt.close()
    if not trades.empty:
        plt.figure(figsize=(8, 4))
        plt.hist(trades["net_return"], bins=40)
        plt.xlabel("Net trade return")
        plt.ylabel("Trades")
        plt.title("Part 5 baseline net trade-return distribution")
        plt.tight_layout()
        plt.savefig(destination / "trade_return_distribution.png", dpi=150)
        plt.close()
    if not daily.empty:
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(daily)), daily["net_pnl"])
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Validation-day sequence")
        plt.ylabel("Daily net P&L")
        plt.title("Part 5 baseline daily net P&L (development)")
        plt.tight_layout()
        plt.savefig(destination / "daily_pnl.png", dpi=150)
        plt.close()


def _load_window(root: Path, name: str, validation_days: tuple[int, ...]) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, str], dict[int, int]]:
    predictions: list[pd.DataFrame] = []
    prices: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    session_seconds: dict[int, int] = {}
    for day in validation_days:
        prediction_path = root / "results/ml/temporal_robustness" / name / "predictions" / f"day{day}.parquet"
        price_path = root / "data/processed" / f"day{day}.parquet"
        prediction = pd.read_parquet(prediction_path)
        price = load_price_day(root, day)
        predictions.append(prediction)
        prices.append(price)
        hashes[f"{name}/prediction_day{day}.parquet"] = sha256_file(prediction_path)
        hashes[f"{name}/price_day{day}.parquet"] = sha256_file(price_path)
        session_seconds[day] = len(price) - 1
    return predictions, prices, hashes, session_seconds


def _summary_with_rows(summary: dict[str, object], rows: int, strategy_name: str | None = None) -> dict[str, object]:
    result = dict(summary)
    result["validation_rows"] = int(rows)
    if strategy_name is not None:
        result["strategy"] = strategy_name
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    scope = audited_scope(root / "results/freeze/development_freeze.json")
    validate_temporal_windows(TEMPORAL_WINDOWS, scope)
    strategy = StrategyConfig()
    strategy.validate()
    costs = TransactionCostModel()
    output = root / "results/ml/backtest_baseline"
    output.mkdir(parents=True, exist_ok=True)
    top_temporal_manifest = json.loads((root / "results/ml/temporal_robustness/run_manifest.json").read_text())
    if top_temporal_manifest.get("holdout_days_loaded") != [] or top_temporal_manifest.get("holdout_accessed"):
        raise RuntimeError("validated temporal artifacts do not prove holdout isolation")

    primary_trades: list[pd.DataFrame] = []
    random_trades_all: list[pd.DataFrame] = []
    passive_trades_all: list[pd.DataFrame] = []
    primary_daily: list[pd.DataFrame] = []
    random_daily_rows: list[pd.DataFrame] = []
    passive_daily_rows: list[pd.DataFrame] = []
    zero_daily_rows: list[pd.DataFrame] = []
    window_metrics: list[dict[str, object]] = []
    baseline_metrics: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}
    random_generator = np.random.default_rng(SEED)
    window_manifests: dict[str, dict[str, object]] = {}

    for name in ("W1", "W2", "W3"):
        train_days = tuple(TEMPORAL_WINDOWS[name]["training_days"])
        validation_days = tuple(TEMPORAL_WINDOWS[name]["validation_days"])
        validate_backtest_scope(scope, validation_days)
        pred_frames, price_frames, hashes, session_seconds = _load_window(root, name, validation_days)
        input_hashes.update(hashes)
        primary_by_day: list[pd.DataFrame] = []
        random_by_day: list[pd.DataFrame] = []
        passive_by_day: list[pd.DataFrame] = []
        for day, prediction, price in zip(validation_days, pred_frames, price_frames):
            primary_by_day.append(simulate_day(prediction, price, window=name, day=day, strategy=strategy, costs=costs))
            random_by_day.append(simulate_day(prediction, price, window=name, day=day, strategy=strategy, costs=costs, direction_mode="random", random_generator=random_generator))
            passive_by_day.append(simulate_passive_day(price, window=name, day=day, strategy=strategy, costs=costs))
        primary_window_trades = pd.concat(primary_by_day, ignore_index=True) if primary_by_day else pd.DataFrame()
        random_window_trades = pd.concat(random_by_day, ignore_index=True) if random_by_day else pd.DataFrame()
        passive_window_trades = pd.concat(passive_by_day, ignore_index=True) if passive_by_day else pd.DataFrame()
        primary_window_daily = daily_pnl_from_trades(primary_window_trades, days=validation_days, window=name, session_seconds=session_seconds)
        random_window_daily = daily_pnl_from_trades(random_window_trades, days=validation_days, window=name, session_seconds=session_seconds)
        passive_window_daily = daily_pnl_from_trades(passive_window_trades, days=validation_days, window=name, session_seconds=session_seconds)
        zero_window_daily = daily_pnl_from_trades(pd.DataFrame(), days=validation_days, window=name, session_seconds=session_seconds)
        primary_trades.append(primary_window_trades)
        random_trades_all.append(random_window_trades)
        passive_trades_all.append(passive_window_trades)
        primary_daily.append(primary_window_daily)
        random_daily_rows.append(random_window_daily)
        passive_daily_rows.append(passive_window_daily)
        zero_daily_rows.append(zero_window_daily)
        primary_summary = _summary_with_rows(summarize_trades(primary_window_trades, primary_window_daily, window=name, strategy=strategy), sum(len(frame) for frame in pred_frames))
        window_metrics.append(primary_summary)
        for label, trades, daily in (("random_direction_null", random_window_trades, random_window_daily), ("passive_long", passive_window_trades, passive_window_daily)):
            baseline_metrics.append(_summary_with_rows(summarize_trades(trades, daily, window=name, strategy=strategy), sum(len(frame) for frame in pred_frames), label))
        zero_summary = _summary_with_rows(summarize_trades(pd.DataFrame(), zero_window_daily, window=name, strategy=strategy), sum(len(frame) for frame in pred_frames), "zero_trade")
        zero_summary["average_exposure"] = 0.0
        zero_summary["maximum_exposure"] = 0.0
        baseline_metrics.append(zero_summary)
        window_manifests[name] = {
            "window": name,
            "training_days": list(train_days),
            "validation_days": list(validation_days),
            "source_days_loaded": list(validation_days),
            "holdout_days_loaded": [],
            "holdout_accessed": False,
            "prediction_source": "validated temporal robustness artifacts",
            "training_only_model_protocol_verified": True,
            "trades": int(len(primary_window_trades)),
            "validation_rows": int(sum(len(frame) for frame in pred_frames)),
            "input_sha256": hashes,
        }
        write_json(window_manifests[name], output / f"{name}_manifest.json")

    all_primary_trades = pd.concat(primary_trades, ignore_index=True)
    all_primary_daily = pd.concat(primary_daily, ignore_index=True)
    all_random_daily = pd.concat(random_daily_rows, ignore_index=True)
    all_passive_daily = pd.concat(passive_daily_rows, ignore_index=True)
    all_zero_daily = pd.concat(zero_daily_rows, ignore_index=True)
    all_primary_equity = equity_curve_from_daily(all_primary_daily, starting_capital=strategy.starting_capital)
    overall = _summary_with_rows(summarize_trades(all_primary_trades, all_primary_daily, window="pooled_development", strategy=strategy), sum(item["validation_rows"] for item in window_metrics))
    random_trades = pd.concat(random_trades_all, ignore_index=True)
    passive_trades = pd.concat(passive_trades_all, ignore_index=True)
    baseline_overall = []
    for label, trades, daily in (("random_direction_null", random_trades, all_random_daily), ("passive_long", passive_trades, all_passive_daily)):
        baseline_overall.append(_summary_with_rows(summarize_trades(trades, daily, window="pooled_development", strategy=strategy), overall["validation_rows"], label))
    zero_overall = _summary_with_rows(summarize_trades(pd.DataFrame(), all_zero_daily, window="pooled_development", strategy=strategy), overall["validation_rows"], "zero_trade")
    zero_overall["average_exposure"] = 0.0
    zero_overall["maximum_exposure"] = 0.0
    baseline_overall.append(zero_overall)

    pd.DataFrame(window_metrics).to_csv(output / "window_metrics.csv", index=False)
    pd.DataFrame(baseline_metrics).to_csv(output / "baseline_metrics.csv", index=False)
    all_primary_trades.to_csv(output / "trade_log.csv", index=False)
    all_primary_daily.to_csv(output / "daily_pnl.csv", index=False)
    all_primary_equity.to_csv(output / "equity_curve.csv", index=False)
    cost_rows = []
    for summary in window_metrics:
        cost_rows.append({"window": summary["window"], "trades": summary["trades"], "gross_pnl": summary["gross_pnl"], "transaction_costs": summary["transaction_costs"], "net_pnl": summary["net_pnl"], "turnover": summary["turnover"], "entry_cost_bps": costs.entry_cost_bps, "exit_cost_bps": costs.exit_cost_bps, "fee_bps": costs.fee_bps, "entry_exit_cost_total": summary["transaction_costs"]})
    pd.DataFrame(cost_rows).to_csv(output / "cost_breakdown.csv", index=False)
    write_json({"primary_strategy": "primary_sign_strategy", "primary_by_window": window_metrics, "overall_development_pooled": overall, "baseline_by_window": baseline_metrics, "baseline_overall": baseline_overall, "holdout_days_loaded": [], "missing_development_days": list(scope.missing_development_days)}, output / "summary_metrics.json")
    write_json({"strategy_name": "primary_sign_strategy", "target_horizon_seconds": 300, "holding_period_seconds": 300, "signal_rule": strategy.signal_rule, "position_rule": strategy.position_rule, "entry_convention": strategy.entry_rule, "exit_convention": strategy.exit_rule, "notional": 1.0, "starting_capital": 1.0, "max_exposure": 1.0, "cost_model": {"entry_cost_bps": costs.entry_cost_bps, "exit_cost_bps": costs.exit_cost_bps, "fee_bps": costs.fee_bps, "spread_slippage_assumption": "5 bps per side combined execution cost", "cost_provenance": "challenge-supported parameterized baseline; no empirical bid/ask/fee fields available"}, "null_baselines": ["zero_trade", "random_direction_null", "passive_long"]}, output / "strategy_config.json")

    sensitivity_trades = all_primary_trades[(all_primary_trades["window"] == "W3") & (all_primary_trades["day"] != 84)]
    sensitivity_daily = all_primary_daily[(all_primary_daily["window"] == "W3") & (all_primary_daily["day"] != 84)].reset_index(drop=True)
    sensitivity = _summary_with_rows(summarize_trades(sensitivity_trades, sensitivity_daily, window="W3_excluding_day84", strategy=strategy), sum(len(frame) for frame in [pd.read_parquet(root / "results/ml/temporal_robustness/W3/predictions" / f"day{day}.parquet") for day in TEMPORAL_WINDOWS["W3"]["validation_days"] if day != 84]))
    normal_w3 = window_metrics[2]
    diff = {key: float(sensitivity[key]) - float(normal_w3[key]) for key in ("gross_pnl", "transaction_costs", "net_pnl", "sharpe", "sortino", "maximum_drawdown", "turnover", "win_rate", "average_trade_return", "median_trade_return", "trade_return_std", "daily_pnl_std", "average_exposure", "maximum_exposure", "trades") if sensitivity.get(key) is not None and normal_w3.get(key) is not None}
    write_json({"diagnostic": "post-hoc W3 validation aggregation excluding Day 84", "retrained": False, "feature_selection_changed": False, "strategy_changed": False, "normal_w3": normal_w3, "excluding_day84": sensitivity, "difference_excluding_day84_minus_normal": diff}, output / "day84_sensitivity.json")

    input_paths = {"development_freeze.json": root / "results/freeze/development_freeze.json", "temporal_run_manifest.json": root / "results/ml/temporal_robustness/run_manifest.json", "config.yaml": root / "config/config.yaml"}
    write_json({"phase": "ML Phase 4 — Part 5 Baseline Strategy + Development Backtest", "created_utc": datetime.now(timezone.utc).isoformat(), "expected_development_days": scope.expected_development_days, "available_development_days": len(scope.available_development_days), "missing_development_days": list(scope.missing_development_days), "windows": {name: {key: list(value) for key, value in window.items()} for name, window in TEMPORAL_WINDOWS.items()}, "source_days_loaded": sorted({day for window in TEMPORAL_WINDOWS.values() for day in window["validation_days"]}), "holdout_days_loaded": [], "holdout_accessed": False, "primary_strategy_only": True, "starting_capital": 1.0, "model_predictions_reused_from_validated_temporal_artifacts": True, "training_only_model_protocol_verified": True, "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()}, "price_and_prediction_input_sha256": input_hashes, "frozen_artifacts_modified": False}, output / "run_manifest.json")
    write_json({"deterministic": True, "random_null_seed": SEED, "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()}, "holdout_days_loaded": [], "primary_strategy": "primary_sign_strategy"}, output / "reproducibility.json")
    _write_figures(root / "figures", all_primary_daily, all_primary_trades, all_primary_equity)
    print(json.dumps({"phase": "ML Phase 4 — Part 5 Baseline Strategy + Development Backtest", "windows": pd.DataFrame(window_metrics)[["window", "trades", "gross_pnl", "transaction_costs", "net_pnl", "sharpe", "maximum_drawdown"]].to_dict("records"), "overall_net_pnl_pooled": overall["net_pnl"], "holdout_days_loaded": []}, indent=2))


if __name__ == "__main__":
    main()
