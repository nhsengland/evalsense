from datetime import date, datetime
from decimal import Decimal
import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset
import numpy as np

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.tasks import DefaultTaskPreprocessor


class TaskValueValidationTests(unittest.TestCase):
    def test_nonfinite_errors_identify_the_nested_field_and_row(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                manager = HuggingFaceDatasetManager(
                    "fixture", version="v1", splits=["test"], data_dir=tmp
                )
                source = Dataset.from_list(
                    [
                        {
                            "input": "private input",
                            "target": "private target",
                            "metadata": {"references": [1e-11, value]},
                        }
                    ]
                )
                with self.assertRaises(ValueError) as caught:
                    DefaultTaskPreprocessor()(source, manager)
                message = str(caught.exception)
                self.assertIn('row 1, field $["metadata"]["references"][1]', message)
                self.assertIn("non-finite float", message)
                self.assertIn("preprocessing function", message)
                self.assertNotIn("private", message)

    def test_older_arrow_nonfinite_scalars_report_the_nested_field(self):
        for dtype in (np.float16, np.float32):
            for value in (math.nan, math.inf, -math.inf):
                with (
                    self.subTest(dtype=dtype, value=value),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    manager = HuggingFaceDatasetManager(
                        "fixture", version="v1", splits=["test"], data_dir=tmp
                    )
                    records = [
                        {
                            "input": "number",
                            "target": "value",
                            "metadata": {"values": (dtype(0.125), dtype(value))},
                        }
                    ]
                    source = Dataset.from_list(records)
                    batch = SimpleNamespace(to_pylist=lambda: records)
                    with patch.object(Dataset, "iter", return_value=iter([batch])):
                        with self.assertRaises(ValueError) as caught:
                            DefaultTaskPreprocessor()(source, manager)
                    self.assertIn(
                        'row 1, field $["metadata"]["values"][1]',
                        str(caught.exception),
                    )
                    self.assertEqual(
                        list((manager.version_path / "tasks").iterdir()), []
                    )

    def test_unsupported_arrow_values_are_not_silently_converted(self):
        values = [
            Decimal("1.234567890123456789"),
            date(2026, 1, 1),
            datetime(2026, 1, 1, 12, 34, 56),
            b"private bytes",
        ]
        for value in values:
            with (
                self.subTest(value_type=type(value)),
                tempfile.TemporaryDirectory() as tmp,
            ):
                manager = HuggingFaceDatasetManager(
                    "fixture", version="v1", splits=["test"], data_dir=tmp
                )
                source = Dataset.from_dict(
                    {
                        "input": ["private input"],
                        "target": ["private target"],
                        "metadata": [{"value": value}],
                    }
                )
                with self.assertRaises(TypeError) as caught:
                    DefaultTaskPreprocessor()(source, manager)
                message = str(caught.exception)
                self.assertIn('row 1, field $["metadata"]["value"]', message)
                self.assertIn(f"unsupported type {type(value).__name__}", message)
                self.assertIn("preprocessing function", message)
                self.assertNotIn("private", message)
                self.assertNotIn(str(value), message)
                self.assertEqual(list((manager.version_path / "tasks").iterdir()), [])

    def test_row_numbers_continue_across_arrow_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="v1", splits=["test"], data_dir=tmp
            )
            source = Dataset.from_dict(
                {"input": ["number"] * 1001, "target": [1e-11] * 1000 + [math.inf]}
            )
            with self.assertRaisesRegex(ValueError, r'row 1001, field \$\["target"\]'):
                DefaultTaskPreprocessor()(source, manager)
            self.assertEqual(list((manager.version_path / "tasks").iterdir()), [])

    def test_field_names_are_unambiguous_and_numeric_strings_remain_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="v1", splits=["test"], data_dir=tmp
            )
            source = Dataset.from_dict(
                {
                    "input": ["number"],
                    "target": ["NaN"],
                    "metadata": [{"not.a.path": {"values": [1e-11, math.nan]}}],
                }
            )
            with self.assertRaises(ValueError) as caught:
                DefaultTaskPreprocessor()(source, manager)
            self.assertIn(
                '$["metadata"]["not.a.path"]["values"][1]', str(caught.exception)
            )
            source = Dataset.from_dict(
                {
                    "input": ["number"],
                    "target": ["NaN"],
                    "metadata": [{"values": ["Infinity", "-Infinity", "1e-11"]}],
                }
            )
            sample = DefaultTaskPreprocessor()(source, manager)[0]
            self.assertEqual(sample.target, "NaN")
            self.assertEqual(
                sample.metadata, {"values": ["Infinity", "-Infinity", "1e-11"]}
            )


if __name__ == "__main__":
    unittest.main()
