import json

import pandas as pd

from src.ebx.ml.dataset_builder import build_model_dataset
from src.ebx.ml.schemas import DevelopmentScope
from src.ebx.ml.splits import chronological_split
from src.ebx.ml.validation import validate_partition


def test_dataset_builder_is_daywise_and_writes_train_validation_partitions(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    for day, base in ((1, 100.0), (2, 110.0)):
        frame = pd.DataFrame({
            "Time": ["00:00:00", "00:00:01", "00:00:02", "00:00:03"],
            "Price": [base, base + 1, base + 2, base + 3],
            "PB1_T1": [1.0, 2.0, 3.0, 4.0],
        })
        frame.to_parquet(processed / f"day{day}.parquet", index=False)
    freeze = tmp_path / "freeze.json"
    aggregate = tmp_path / "aggregate.csv"
    config = tmp_path / "config.yaml"
    freeze.write_text("freeze")
    aggregate.write_text("aggregate")
    config.write_text("config")
    scope = DevelopmentScope(85, (1, 2), tuple(range(3, 80)), tuple(range(86, 109)))
    split = chronological_split(scope, validation_start_day=2)
    frozen = pd.DataFrame({"feature": ["PB1_T1"], "horizon_seconds": [1], "eligible_for_ml": [True]})
    result = build_model_dataset(
        processed_dir=processed,
        output_root=tmp_path / "ml",
        scope=scope,
        split=split,
        frozen_screen=frozen,
        target_horizon=1,
        frozen_paths={"freeze": freeze, "aggregate": aggregate, "config": config},
    )
    assert result["dataset_manifest"]["row_count"] == 6
    train_path = tmp_path / "ml/datasets/train/day1.parquet"
    validation_path = tmp_path / "ml/datasets/validation/day2.parquet"
    assert validate_partition(train_path, (1, 2), ("PB1_T1",))["rows"] == 3
    assert validate_partition(validation_path, (1, 2), ("PB1_T1",))["rows"] == 3
    manifest = json.loads((tmp_path / "ml/preprocessing/preprocessing_manifest.json").read_text())
    assert manifest["fit_days"] == [1]
    assert manifest["validation_days_not_used_for_fit"] == [2]
