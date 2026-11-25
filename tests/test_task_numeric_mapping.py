"""Numerical values survive dataset loading and Inspect sample mapping."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from datasets import Dataset, DatasetDict, Features, Value
from inspect_ai.dataset import FieldSpec, Sample

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.tasks import DefaultTaskPreprocessor, TaskPreprocessor


def numeric_target(sample: Sample) -> float:
    assert isinstance(sample.target, str)
    return float(sample.target)


class TaskNumericMappingTests(unittest.TestCase):
    def test_saved_dataset_reaches_field_spec_with_exact_targets_and_metadata(self):
        values = [1e-11, -3e-11, 1.2345678901234567, 123.123456789012, -0.0]
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="1", splits=["test"], data_dir=tmp
            )
            source = Dataset.from_dict(
                {
                    "question": ["measurement"] * len(values),
                    "answer": values,
                    "measurement": [
                        {"reference": value, "bounds": [value, -value]}
                        for value in values
                    ],
                }
            )
            DatasetDict({"test": source}).save_to_disk(manager.main_data_path)
            loaded = manager.load(retrieve=False)
            self.assertEqual(
                [value.hex() for value in loaded["answer"]], [v.hex() for v in values]
            )

            samples = DefaultTaskPreprocessor()(
                loaded,
                manager,
                field_spec=FieldSpec(
                    input="question", target="answer", metadata=["measurement"]
                ),
            )
            for expected, sample in zip(values, samples, strict=True):
                self.assertEqual(numeric_target(sample).hex(), expected.hex())
                assert sample.metadata is not None
                measurement = sample.metadata["measurement"]
                self.assertEqual(measurement["reference"].hex(), expected.hex())
                self.assertEqual(
                    [value.hex() for value in measurement["bounds"]],
                    [expected.hex(), (-expected).hex()],
                )
            assert samples.location is not None
            self.assertEqual(
                Path(samples.location).parent, manager.version_path / "tasks"
            )
            self.assertEqual(samples.name, Path(samples.location).stem)

    def test_record_mapper_receives_exact_values_before_filtering_and_fanout(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="1", splits=["test"], data_dir=tmp
            )
            values = [1e-11, -2e-11, 1.2345678901234567]
            source = Dataset.from_dict(
                {
                    "input": ["keep", "drop", "twice"],
                    "target": values,
                    "metadata": [
                        {"reference": value, "bounds": [value, -value]}
                        for value in values
                    ],
                }
            )
            received = []

            def mapper(row):
                received.append(row)
                if row["input"] == "drop":
                    return []
                count = 2 if row["input"] == "twice" else 1
                return [
                    Sample(
                        input=f"{row['input']}-{i}",
                        target=str(row["target"]),
                        metadata=row["metadata"],
                    )
                    for i in range(count)
                ]

            result = DefaultTaskPreprocessor()(source, manager, field_spec=mapper)
            self.assertEqual(
                [row["target"].hex() for row in received], [v.hex() for v in values]
            )
            self.assertEqual(
                [s.input for s in result], ["keep-0", "twice-0", "twice-1"]
            )
            for sample, expected in zip(
                result, [values[0], values[2], values[2]], strict=True
            ):
                self.assertEqual(numeric_target(sample).hex(), expected.hex())
                assert sample.metadata is not None
                self.assertEqual(sample.metadata["reference"].hex(), expected.hex())

    def test_float32_values_retain_the_precision_in_the_source_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="1", splits=["test"], data_dir=tmp
            )
            values = [1e-11, -3e-11, 1.23456789]
            source = Dataset.from_dict(
                {
                    "input": ["measurement"] * len(values),
                    "target": values,
                    "metadata": [{"reference": value} for value in values],
                },
                features=Features(
                    {
                        "input": Value("string"),
                        "target": Value("float32"),
                        "metadata": {"reference": Value("float32")},
                    }
                ),
            )
            stored = source["target"]
            self.assertNotEqual(stored[0], values[0])
            result = DefaultTaskPreprocessor()(source, manager)
            for sample, expected in zip(result, stored, strict=True):
                self.assertEqual(numeric_target(sample).hex(), expected.hex())
                assert sample.metadata is not None
                self.assertEqual(sample.metadata["reference"].hex(), expected.hex())

    def test_selected_row_order_and_all_columns_survive_display_formats(self):
        source = Dataset.from_dict(
            {
                "id": ["first", "second", "third"],
                "input": ["one", "two", "three"],
                "target": [1e-11, -2e-11, 1.2345678901234567],
                "metadata": [{"active": True}, {"active": False}, {"active": True}],
            }
        ).select([2, 0, 2, 1])
        for format_name in (None, "numpy", "pandas", "arrow"):
            with self.subTest(format=format_name), tempfile.TemporaryDirectory() as tmp:
                manager = HuggingFaceDatasetManager(
                    "fixture", version="1", splits=["test"], data_dir=tmp
                )
                formatted = source.with_format(format_name, columns=["target"])
                original_format = deepcopy(formatted.format)
                result = DefaultTaskPreprocessor()(formatted, manager)
                self.assertEqual(formatted.format, original_format)
                self.assertEqual(
                    [s.id for s in result], ["third", "first", "third", "second"]
                )
                self.assertEqual(
                    [s.input for s in result], ["three", "one", "three", "two"]
                )
                self.assertEqual(
                    [numeric_target(s).hex() for s in result],
                    [
                        v.hex()
                        for v in [1.2345678901234567, 1e-11, 1.2345678901234567, -2e-11]
                    ],
                )
                self.assertEqual([s.metadata for s in result], list(source["metadata"]))

    def test_materialised_preprocessing_changes_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="1", splits=["test"], data_dir=tmp
            )
            source = Dataset.from_dict(
                {"input": ["first", "second"], "target": [1e-11, -2e-11]}
            )
            calls = []

            def preprocess(hf_dataset, dataset_manager):
                calls.append(hf_dataset)
                return hf_dataset.map(
                    lambda row: {
                        "target": row["target"] * 3,
                        "metadata": {"original": row["target"]},
                    }
                )

            pre = TaskPreprocessor("scaled", preprocess, version="1")
            for _ in range(2):
                result = pre(source, manager)
                for sample, original in zip(result, source["target"], strict=True):
                    self.assertEqual(numeric_target(sample).hex(), (original * 3).hex())
                    assert sample.metadata is not None
                    self.assertEqual(sample.metadata["original"].hex(), original.hex())
            self.assertEqual(len(calls), 1)

    def test_lazy_display_transform_is_not_applied_during_task_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="1", splits=["test"], data_dir=tmp
            )
            calls = []

            def display_transform(batch):
                calls.append(batch)
                raise AssertionError(
                    "Display transform must not alter stored task data"
                )

            source = Dataset.from_dict(
                {"input": ["measurement"], "target": [1e-11]}
            ).with_transform(display_transform)
            original_format = deepcopy(source.format)
            result = DefaultTaskPreprocessor()(source, manager)
            self.assertEqual(numeric_target(result[0]).hex(), (1e-11).hex())
            self.assertEqual(source.format, original_format)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
