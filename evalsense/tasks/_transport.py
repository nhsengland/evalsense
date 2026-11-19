"""JSONL transport for task records without decimal-precision rounding."""

import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from datasets import Dataset
import numpy as np
from pyarrow import Table


TASK_DATA_FORMAT = "evalsense.task-jsonl.v1"


def _encode_arrow_scalar(value: object) -> float:
    # Older Arrow versions return NumPy half scalars from to_pylist(). Binary16
    # and binary32 values promote exactly; other unsupported types remain errors.
    if type(value) in (np.float16, np.float32):
        return float(cast(np.floating, value))
    raise TypeError(f"Unsupported task value type: {type(value).__name__}")


def _validate_value(value: object, row_number: int, field_path: str) -> None:
    """Identify unsupported values without including record contents in errors."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) or type(value) in (np.float16, np.float32):
        if math.isfinite(float(cast(float | np.floating, value))):
            return
        raise ValueError(
            f"Task data row {row_number}, field {field_path} contains a non-finite "
            "float. Convert or remove it in the preprocessing function."
        )
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_value(item, row_number, f"{field_path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Task data row {row_number}, field {field_path} has a "
                    f"{type(key).__name__} key. Convert keys to strings in the "
                    "preprocessing function."
                )
            _validate_value(item, row_number, f"{field_path}[{json.dumps(key)}]")
        return
    raise TypeError(
        f"Task data row {row_number}, field {field_path} has unsupported type "
        f"{type(value).__name__}. Convert or remove it in the preprocessing "
        "function; task data must contain finite JSON values."
    )


def write_task_data(dataset: Dataset, path: Path) -> None:
    """Write finite JSON values atomically, retaining their source precision.

    Read stored Arrow values, as Dataset.to_json does, without applying display
    formats, lazy transforms or feature decoding. Unsupported values raise rather
    than being coerced to strings, rounded numbers or nulls.
    """
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            row_number = 0
            for batch in dataset.with_format("arrow").iter(batch_size=1000):
                for record in cast(Table, batch).to_pylist():
                    row_number += 1
                    _validate_value(record, row_number, "$")
                    stream.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            allow_nan=False,
                            default=_encode_arrow_scalar,
                        )
                    )
                    stream.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
