"""JSONL transport for task records without decimal-precision rounding."""

import json
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
            for batch in dataset.with_format("arrow").iter(batch_size=1000):
                for record in cast(Table, batch).to_pylist():
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
