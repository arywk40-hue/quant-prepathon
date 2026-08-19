import pandas as pd

from src.ebx.ml.baseline import RidgeBaseline, validation_metrics


def test_baseline_reads_only_explicit_train_and_validation_partitions(tmp_path):
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    train_dir.mkdir()
    validation_dir.mkdir()
    for directory, day, base in ((train_dir, 1, 0.0), (validation_dir, 80, 1.0)):
        pd.DataFrame({
            "day": [day] * 4,
            "timestamp": ["00:00:00", "00:00:01", "00:00:02", "00:00:03"],
            "timestamp_seconds": [0, 1, 2, 3],
            "target": [base, base + 1, base + 2, base + 3],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [4.0, 3.0, 2.0, 1.0],
        }).to_parquet(directory / f"day{day}.parquet", index=False)
    model = RidgeBaseline(("f1", "f2"), alpha=1.0).fit_partition_paths([train_dir / "day1.parquet"])
    validation = pd.read_parquet(validation_dir / "day80.parquet")
    output = validation[["day", "target"]].copy()
    output["prediction"] = model.predict(validation)
    pooled, daily = validation_metrics(output)
    assert pooled["validation_observations"] == 4
    assert daily.iloc[0]["day"] == 80
    assert not list(tmp_path.rglob("day86.parquet"))
