"""One-day CSV loader and Parquet writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.ingestion.discovery import parse_day_filename

try:
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as papq
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    pa = None
    pacsv = None
    papq = None
    _PYARROW_IMPORT_ERROR = exc
else:
    _PYARROW_IMPORT_ERROR = None


@dataclass(frozen=True)
class LoadedDay:
    day: int
    source_path: Path
    table: "pa.Table"

    @property
    def rows(self) -> int:
        return self.table.num_rows

    @property
    def columns(self) -> list[str]:
        return list(self.table.column_names)


def require_pyarrow() -> None:
    if pa is None:
        raise RuntimeError(
            "PyArrow is required for Phase 2 CSV loading and Parquet output"
        ) from _PYARROW_IMPORT_ERROR


def load_day(path: Path, day: int) -> LoadedDay:
    """Load one CSV without sorting, imputation, or cross-day concatenation."""

    require_pyarrow()
    file_day = parse_day_filename(path.name)
    if file_day is None or file_day != day:
        raise ValueError(f"day ID {day} does not match source filename {path.name}")
    read_options = pacsv.ReadOptions(use_threads=True, block_size=1 << 20)
    parse_options = pacsv.ParseOptions(delimiter=",")
    # Keep the source timestamp representation exact. If Arrow infers Time as
    # time32[s], Parquet legally normalizes it to milliseconds and a strict
    # round-trip schema check fails despite identical displayed timestamps.
    convert_options = pacsv.ConvertOptions(
        null_values=[""],
        strings_can_be_null=True,
        column_types={"Time": pa.string()},
    )
    table = pacsv.read_csv(
        str(path),
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )
    return LoadedDay(day=day, source_path=path, table=table)


def schema_record(table: "pa.Table") -> list[dict[str, str]]:
    return [{"name": field.name, "type": str(field.type)} for field in table.schema]


def write_parquet(table: "pa.Table", path: Path, compression: str = "zstd") -> dict[str, object]:
    require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    papq.write_table(table, str(path), compression=compression)
    roundtrip = papq.read_table(str(path))
    return {
        "rows": roundtrip.num_rows,
        "columns": list(roundtrip.column_names),
        "schema_equal": roundtrip.schema == table.schema,
        "values_equal": roundtrip.equals(table),
        "path": str(path),
    }
