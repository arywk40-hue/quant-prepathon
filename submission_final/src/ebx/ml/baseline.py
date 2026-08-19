"""Minimal deterministic Ridge baseline for the frozen ML Phase 0 dataset."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import DevelopmentScope


@dataclass
class RidgeBaseline:
    """Ridge regression fitted from day-wise sufficient statistics.

    The fit is equivalent to regularized least squares with an unpenalized
    intercept. It avoids concatenating all training partitions in memory and
    has no random state or hyperparameter search.
    """

    feature_names: tuple[str, ...]
    alpha: float = 1.0
    fit_intercept: bool = True
    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    n_train_samples_: int = 0
    gram_condition_number_: float | None = None

    def fit_partition_paths(self, paths: list[str | Path]) -> "RidgeBaseline":
        if self.alpha < 0:
            raise ValueError("Ridge alpha must be non-negative")
        if not paths:
            raise ValueError("at least one training partition is required")
        feature_count = len(self.feature_names)
        dimension = feature_count + int(self.fit_intercept)
        gram = np.zeros((dimension, dimension), dtype=np.float64)
        rhs = np.zeros(dimension, dtype=np.float64)
        count = 0
        columns = [*self.feature_names, "target"]
        for path in paths:
            frame = pd.read_parquet(path, columns=columns)
            x = frame.loc[:, list(self.feature_names)].to_numpy(dtype=np.float64)
            y = frame["target"].to_numpy(dtype=np.float64)
            if len(frame) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"invalid training partition: {path}")
            gram[:feature_count, :feature_count] += x.T @ x
            rhs[:feature_count] += x.T @ y
            if self.fit_intercept:
                feature_sum = x.sum(axis=0)
                gram[:feature_count, feature_count] += feature_sum
                gram[feature_count, :feature_count] += feature_sum
                gram[feature_count, feature_count] += len(frame)
                rhs[feature_count] += y.sum()
            count += len(frame)
        gram[:feature_count, :feature_count] += self.alpha * np.eye(feature_count)
        try:
            solution = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Ridge normal equations could not be solved") from exc
        self.coef_ = solution[:feature_count]
        self.intercept_ = float(solution[feature_count]) if self.fit_intercept else 0.0
        self.n_train_samples_ = count
        self.gram_condition_number_ = float(np.linalg.cond(gram))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Ridge baseline has not been fitted")
        if tuple(frame.columns) == self.feature_names:
            features = frame
        else:
            missing = set(self.feature_names) - set(frame.columns)
            if missing:
                raise ValueError(f"prediction frame is missing features: {sorted(missing)}")
            features = frame.loc[:, list(self.feature_names)]
        values = features.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("prediction features contain non-finite values")
        return values @ self.coef_ + self.intercept_

    def summary(self) -> dict[str, object]:
        if self.coef_ is None:
            raise ValueError("Ridge baseline has not been fitted")
        return {
            "model": "ridge",
            "alpha": float(self.alpha),
            "fit_intercept": bool(self.fit_intercept),
            "feature_count": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "n_train_samples": int(self.n_train_samples_),
            "intercept": float(self.intercept_),
            "coefficient_l2_norm": float(np.linalg.norm(self.coef_)),
            "normal_equation_condition_number": self.gram_condition_number_,
        }

    def save(self, path: str | Path) -> None:
        if self.coef_ is None:
            raise ValueError("cannot save an unfitted Ridge baseline")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle, protocol=5)

    @classmethod
    def load(cls, path: str | Path) -> "RidgeBaseline":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("model artifact is not a RidgeBaseline")
        return model


def validate_baseline_scope(
    scope: DevelopmentScope,
    training_days: list[int],
    validation_days: list[int],
    feature_names: tuple[str, ...],
    target_horizon: int,
) -> None:
    """Reject wrong split, feature count, target horizon, or holdout scope."""

    scope.assert_development_days(training_days + validation_days)
    if set(training_days) & set(validation_days):
        raise ValueError("training and validation days overlap")
    if max(training_days) >= min(validation_days):
        raise ValueError("baseline split is not chronological")
    if len(feature_names) != 197:
        raise ValueError(f"expected the frozen 197-feature set, got {len(feature_names)}")
    if target_horizon != 300:
        raise ValueError("Phase 3 baseline target must be 300 seconds")
    if set(training_days + validation_days) != set(scope.available_development_days):
        raise ValueError("baseline split does not cover the available development days")


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _corr(
        pd.Series(left).rank(method="average").to_numpy(dtype=float),
        pd.Series(right).rank(method="average").to_numpy(dtype=float),
    )


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((target - target.mean()) ** 2))
    return float(1.0 - np.sum((target - prediction) ** 2) / denominator) if denominator else float("nan")


def _metric_row(day: int | str, target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    if len(target) != len(prediction):
        raise ValueError("prediction and target lengths differ")
    error = prediction - target
    return {
        "day": day,
        "validation_observations": int(len(target)),
        "pearson_ic": _corr(prediction, target),
        "spearman_ic": _spearman(prediction, target),
        "directional_accuracy": float(np.mean(np.sign(prediction) == np.sign(target))),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction, ddof=1)) if len(prediction) > 1 else float("nan"),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target, ddof=1)) if len(target) > 1 else float("nan"),
        "r2": _r2(target, prediction),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


def validation_metrics(predictions: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    """Calculate pooled and daily validation metrics from aligned rows."""

    required = {"day", "target", "prediction"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"predictions missing columns: {sorted(required - set(predictions.columns))}")
    daily = pd.DataFrame([
        _metric_row(day, group["target"].to_numpy(dtype=float), group["prediction"].to_numpy(dtype=float))
        for day, group in predictions.groupby("day", sort=True)
    ])
    target = predictions["target"].to_numpy(dtype=float)
    prediction = predictions["prediction"].to_numpy(dtype=float)
    pooled = _metric_row("pooled", target, prediction)
    daily_ic = daily["pearson_ic"].dropna().to_numpy(dtype=float)
    daily_spearman = daily["spearman_ic"].dropna().to_numpy(dtype=float)
    pooled.update({
        "mean_daily_pearson_ic": float(np.mean(daily_ic)) if len(daily_ic) else float("nan"),
        "median_daily_pearson_ic": float(np.median(daily_ic)) if len(daily_ic) else float("nan"),
        "std_daily_pearson_ic": float(np.std(daily_ic, ddof=1)) if len(daily_ic) > 1 else float("nan"),
        "mean_daily_spearman_ic": float(np.mean(daily_spearman)) if len(daily_spearman) else float("nan"),
        "median_daily_spearman_ic": float(np.median(daily_spearman)) if len(daily_spearman) else float("nan"),
        "std_daily_spearman_ic": float(np.std(daily_spearman, ddof=1)) if len(daily_spearman) > 1 else float("nan"),
        "validation_days": int(len(daily)),
    })
    return pooled, daily
