# ML Phase 4 — Part 5 Baseline Strategy and Development Backtest

## Scope and protocol

This is one pre-specified economic-utility test of the already validated temporal Ridge predictions. No model was retrained and no strategy parameter was selected after observing results.

| Item | Value |
|---|---|
| Expected development days | 85 |
| Available development days | 70 |
| Missing development days | 65–79 (15 days; retained as explicit gaps) |
| Development validation days processed | 45–64 and 80–85 (26 validation days) |
| Holdout days loaded | None (`[]`) |
| Holdout days | 86–108; untouched |
| Target / holding period | 300 seconds |
| Model predictions | Reused from validated Phase 3 temporal Ridge artifacts |
| Training windows | W1: 1–44; W2: 1–54; W3: 1–64 |
| Validation windows | W1: 45–54; W2: 55–64; W3: 80–85 |

The temporal artifacts’ manifests were checked for training-only selection/preprocessing and `holdout_days_loaded: []`. Part 5 did not use `target` to generate signals; it used only timestamped predictions and raw same-day prices for execution accounting.

## Primary strategy

- Prediction > 0: LONG; prediction < 0: SHORT; prediction = 0: FLAT.
- Fixed unit notional and normalized starting capital of 1.0.
- One position at a time. Signals before the exact exit are ignored; a signal at the exit timestamp may open the next position.
- Entry is the first raw price observation at or after the signal timestamp.
- Exit is the exact raw price observation at `entry timestamp + 300 seconds`, within the same day. Missing exact exits are skipped; there is no interpolation and no cross-day position.
- Gross return is `direction × (exit_price / entry_price − 1)`.

No validated bid/ask, fee, or slippage fields exist in the repository. The parameterized baseline therefore charges 5 bps at entry and 5 bps at exit, proportional to notional, with fee 0 bps. This is an explicit challenge-supported assumption, not an empirically calibrated cost model. Turnover is entry plus exit notional.

Sharpe is annualized from daily net P&L using `sqrt(252)`. Sortino uses zero-target downside deviation. Maximum drawdown is measured from normalized equity starting at 1.0.

## Primary results

P&L is in normalized notional units.

| Window | Validation rows | Trades | Gross P&L | Costs | Net P&L | Sharpe | Sortino | Max drawdown | Turnover | Win rate | Avg trade return | Median trade return | Trade return SD | Daily P&L SD | Avg exposure | Max exposure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | 122420 | 410 | 0.008677674912 | 0.410000000000 | -0.401322325088 | -186.162764434 | -15.822818350 | -0.376028894 | 820 | 0.063414634 | -0.000978835 | -0.001000000 | 0.000775000 | 0.003422164 | 0.527014868 | 1.0 |
| W2 | 84632 | 285 | 0.014409137940 | 0.285000000000 | -0.270590862060 | -32.994426769 | -14.441311105 | -0.243738858 | 570 | 0.059649123 | -0.000949442 | -0.000983407 | 0.000592384 | 0.013018856 | 0.413585878 | 1.0 |
| W3 | 73452 | 246 | 0.021281444253 | 0.246000000000 | -0.224718555747 | -106.280913202 | -15.728970520 | -0.191784072 | 492 | 0.085365854 | -0.000913490 | -0.000965588 | 0.000646045 | 0.005594132 | 0.527014868 | 1.0 |

### Daily primary net P&L

| Window | Day | Trades | Gross P&L | Costs | Net P&L |
|---|---:|---:|---:|---:|---:|
| W1 | 45 | 41 | 0.000463778 | 0.041 | -0.040536222 |
| W1 | 46 | 41 | 0.000593451 | 0.041 | -0.040406549 |
| W1 | 47 | 41 | 0.004044264 | 0.041 | -0.036955736 |
| W1 | 48 | 41 | 0.002334220 | 0.041 | -0.038665780 |
| W1 | 49 | 41 | -0.005485511 | 0.041 | -0.046485511 |
| W1 | 50 | 41 | -0.000779060 | 0.041 | -0.041779060 |
| W1 | 51 | 41 | 0.007129701 | 0.041 | -0.033870299 |
| W1 | 52 | 41 | 0.000986304 | 0.041 | -0.040013696 |
| W1 | 53 | 41 | 0.001678971 | 0.041 | -0.039321029 |
| W1 | 54 | 41 | -0.002288443 | 0.041 | -0.043288443 |
| W2 | 55–59 | 41/day | — | — | -0.035506259, -0.039954243, -0.042993172, -0.034986945, -0.039111820 |
| W2 | 60 | 21 | -0.001449920 | 0.021 | -0.022449920 |
| W2 | 61 | 21 | -0.001964141 | 0.021 | -0.022964141 |
| W2 | 62 | 13 | 0.001719876 | 0.013 | -0.011280124 |
| W2 | 63 | 13 | 0.001404009 | 0.013 | -0.011595991 |
| W2 | 64 | 12 | 0.002251755 | 0.012 | -0.009748245 |
| W3 | 80 | 41 | 0.000250390 | 0.041 | -0.040749610 |
| W3 | 81 | 41 | 0.000571272 | 0.041 | -0.040428728 |
| W3 | 82 | 41 | 0.000931636 | 0.041 | -0.040068364 |
| W3 | 83 | 41 | -0.000483567 | 0.041 | -0.041483567 |
| W3 | 84 | 41 | 0.013893638 | 0.041 | -0.027106362 |
| W3 | 85 | 41 | 0.006118075 | 0.041 | -0.034881925 |

The complete day-level table is in `results/ml/backtest_baseline/daily_pnl.csv`.

### Pooled development summary

The descriptive pooled primary result contains 941 trades, gross P&L `0.044368257105`, costs `0.941000000000`, net P&L `-0.896631742895`, Sharpe `-52.575858140`, Sortino `-15.221378104`, maximum drawdown `-0.892264555`, turnover `1882`, win rate `0.068012752`, average trade return `-0.000952850`, and 280,504 prediction rows.

This pooled figure concatenates W1, W2, and W3 validation results, so some calendar days occur in more than one temporal experiment. It is not presented as a unique continuous portfolio result. The window-specific figures above are the primary development evidence.

## Post-hoc Day-84 sensitivity

Day 84 remains in the official W3 primary result. The following is a diagnostic aggregation only; there was no retraining, feature-selection change, or strategy change.

| W3 aggregation | Rows | Trades | Gross P&L | Costs | Net P&L | Sharpe | Sortino | Max drawdown | Turnover | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal, including Day 84 (PRIMARY) | 73452 | 246 | 0.021281444253 | 0.246000000000 | -0.224718555747 | -106.280913202 | -15.728970520 | -0.191784072 | 492 | 0.085365854 |
| Excluding Day 84 (POST-HOC SENSITIVITY) | 61210 | 205 | 0.007387805990 | 0.205000000000 | -0.197612194010 | -237.110360187 | -15.846122554 | -0.163526214 | 410 | 0.058536585 |

The positive gross contribution from Day 84 does not overcome the fixed costs under this baseline; both W3 aggregations have negative net P&L.

## Comparator baselines

The comparators are not candidate strategies and were not used to select or alter the primary rule.

| Window | Comparator | Trades | Gross P&L | Costs | Net P&L | Sharpe | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|
| W1 | Random direction (seed 20260818) | 410 | -0.009336531 | 0.410 | -0.419336531 | -213.673219574 | -0.395673920 |
| W1 | Passive long | 10 | 0.025408047 | 0.010 | 0.015408047 | 5.949086648 | -0.002919796 |
| W2 | Random direction (seed 20260818) | 285 | -0.009492547 | 0.285 | -0.294492547 | -34.418950205 | -0.270358472 |
| W2 | Passive long | 10 | 0.009458382 | 0.010 | -0.000541618 | -0.159716779 | -0.016709667 |
| W3 | Random direction (seed 20260818) | 246 | -0.002583080 | 0.246 | -0.248583080 | -330.372877693 | -0.216927560 |
| W3 | Passive long | 6 | 0.020444332 | 0.006 | 0.014444332 | 5.415770614 | -0.008343750 |

The zero-trade comparator is recorded in `baseline_metrics.csv` and `summary_metrics.json` with zero P&L, zero costs, zero turnover, and zero trades. Passive-long results are descriptive only and are not a claim of investable performance.

## Artifacts and figures

The isolated output namespace is `results/ml/backtest_baseline/` and contains:

- `trade_log.csv`
- `daily_pnl.csv`
- `window_metrics.csv`
- `baseline_metrics.csv`
- `summary_metrics.json`
- `equity_curve.csv`
- `cost_breakdown.csv`
- `strategy_config.json`
- `run_manifest.json`
- `reproducibility.json`
- `W1_manifest.json`, `W2_manifest.json`, and `W3_manifest.json`
- `day84_sensitivity.json`

Figures are under `figures/ml_phase4/`: `equity_curve.png`, `daily_pnl.png`, and `trade_return_distribution.png`.

## Integrity and limitations

- The implementation is in `src/ebx/ml/backtest.py`; the reproducible entry point is `scripts/ml/phase_ml4_backtest.py`.
- No raw CSV, development freeze, holdout freeze, aggregate IC, frozen feature, baseline, train-only selection, temporal robustness, or Day-84 forensic artifact was modified.
- No Days 65–79 were fabricated or loaded. Days 86–108 were not loaded, inspected, or used; manifests explicitly contain `holdout_days_loaded: []`.
- There are no quantity, bid/ask, fee, or contract-value fields in the available schema. P&L is therefore normalized unit-notional P&L, not currency P&L.
- The 5 bps-per-side assumption is parameterized but not empirically validated from the data.
- Development validation days are reused across temporal windows by design. The pooled result is descriptive and must not be interpreted as a production equity curve.
- This is not evidence of profitability, alpha, production readiness, or holdout generalization. No holdout evaluation, final production model, strategy optimization, parameter search, or backtest beyond this baseline was performed.
