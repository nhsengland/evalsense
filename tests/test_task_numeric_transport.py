import json
import math
import sys
import tempfile
import unittest
from typing import cast
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

from datasets import Dataset
import numpy as np
from inspect_ai.dataset import FieldSpec
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import Evaluator, ExperimentConfig
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.tasks import DefaultTaskPreprocessor, TaskPreprocessor
from evalsense.utils.cache import fingerprint
from evalsense.utils.files import to_safe_filename


class TaskNumericTransportTests(unittest.TestCase):
    def test_finite_references_round_trip_without_changing_binary_value(self):
        values = [
            1e-11,
            -3e-11,
            6.6743e-11,
            0.125,
            -3.75,
            1.2345678901234567,
            123.123456789012,
            math.nextafter(1.0, 2.0),
            sys.float_info.min,
            math.ulp(0.0),
            sys.float_info.max,
            -sys.float_info.max,
            0.0,
            -0.0,
        ]
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="v1", splits=["test"], data_dir=tmp
            )
            source = Dataset.from_dict(
                {"input": ["number"] * len(values), "target": values}
            )
            samples = DefaultTaskPreprocessor()(source, manager)
            self.assertEqual(
                [float(cast(str, sample.target)).hex() for sample in samples],
                [value.hex() for value in values],
            )
            paths = list((manager.version_path / "tasks").glob("*.jsonl"))
            self.assertEqual(len(paths), 1)
            records = [json.loads(line) for line in paths[0].read_text().splitlines()]
            self.assertEqual(
                [record["target"].hex() for record in records],
                [value.hex() for value in values],
            )

    def test_older_arrow_numpy_scalars_preserve_their_stored_precision(self):
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                manager = HuggingFaceDatasetManager(
                    "fixture", version="v1", splits=["test"], data_dir=tmp
                )
                values = [dtype(0.3333), dtype(-0.0), np.nextafter(dtype(0), dtype(1))]
                records = [
                    {
                        "input": "number",
                        "target": value,
                        "metadata": {"nested": [value]},
                    }
                    for value in values
                ]
                source = Dataset.from_list(records)
                # Older Arrow releases returned NumPy float16 scalars. Keep
                # this representation covered when running with newer Arrow.
                batch = SimpleNamespace(to_pylist=lambda: records)
                with patch.object(Dataset, "iter", return_value=iter([batch])):
                    samples = DefaultTaskPreprocessor()(source, manager)
                for value, sample in zip(values, samples, strict=True):
                    self.assertEqual(
                        float(cast(str, sample.target)).hex(), float(value).hex()
                    )
                    assert sample.metadata is not None
                    self.assertEqual(
                        sample.metadata["nested"][0].hex(), float(value).hex()
                    )

    def test_json_metadata_keeps_integers_booleans_nulls_and_nested_numbers(self):
        metadata = {
            "large_integer": 2**63 - 1,
            "enabled": True,
            "missing": None,
            "nested": {"references": [1e-11, -1.2345678901234567]},
            "text": "line one\nλ",
        }
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="v1", splits=["test"], data_dir=tmp
            )
            source = Dataset.from_dict(
                {"input": ["number"], "target": ["1e-11"], "metadata": [metadata]}
            )
            sample = DefaultTaskPreprocessor()(source, manager)[0]
            self.assertEqual(sample.target, "1e-11")
            self.assertEqual(sample.metadata, metadata)
            assert sample.metadata is not None
            self.assertIs(type(sample.metadata["large_integer"]), int)
            self.assertIs(type(sample.metadata["enabled"]), bool)

    def test_nonfinite_values_do_not_leave_a_partial_cache(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                manager = HuggingFaceDatasetManager(
                    "fixture", version="v1", splits=["test"], data_dir=tmp
                )
                source = Dataset.from_dict(
                    {"input": ["valid", "invalid"], "target": [1e-11, value]}
                )
                with self.assertRaises(ValueError):
                    DefaultTaskPreprocessor()(source, manager)
                self.assertEqual(list((manager.version_path / "tasks").iterdir()), [])

    def test_old_rounded_task_cache_is_left_intact_and_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="v1", splits=["test"], data_dir=tmp
            )
            preprocessor = DefaultTaskPreprocessor()
            source = Dataset.from_dict({"input": ["number"], "target": [1e-11]})
            old_preprocessor_key = fingerprint(
                {
                    "implementation": type(preprocessor),
                    "function": preprocessor.preprocessing_function,
                },
                version=preprocessor.version,
            )
            old_data_key = fingerprint(
                {
                    "dataset": manager.record,
                    "content": source._fingerprint,
                    "preprocessor": old_preprocessor_key,
                }
            )
            tasks = manager.version_path / "tasks"
            tasks.mkdir(parents=True)
            old_path = tasks / (
                f"{to_safe_filename(preprocessor.name)[:40]}-{old_data_key}.jsonl"
            )
            old_content = '{"input":"number","target":0.0}\n'
            old_path.write_text(old_content)
            samples = preprocessor(source, manager)
            self.assertEqual(float(cast(str, samples[0].target)), 1e-11)
            self.assertEqual(old_path.read_text(), old_content)
            self.assertEqual(len(list(tasks.glob("*.jsonl"))), 2)

    def test_transport_change_separates_generation_and_evaluation_records(self):
        preprocessor = DefaultTaskPreprocessor()
        experiment = ExperimentConfig(
            HuggingFaceDatasetManager("fixture", version="v1", splits=["test"]),
            GenerationSteps("generate", generate()),
            ModelConfig("mockllm/model"),
            evaluator=Evaluator("exact", exact()),
            task_preprocessor=preprocessor,
        )
        old_key = fingerprint(
            {
                "implementation": type(preprocessor),
                "function": preprocessor.preprocessing_function,
            },
            version=preprocessor.version,
        )
        with patch.object(
            TaskPreprocessor,
            "cache_key",
            new_callable=PropertyMock,
            return_value=old_key,
        ):
            old_generation = experiment.generation_record
            old_evaluation = experiment.evaluation_record
        self.assertNotEqual(old_generation, experiment.generation_record)
        self.assertNotEqual(old_evaluation, experiment.evaluation_record)
        self.assertEqual(
            old_generation.generator_fingerprint,
            experiment.generation_record.generator_fingerprint,
        )

    def test_repeat_calls_reuse_precise_data_and_apply_the_current_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="v1", splits=["test"], data_dir=tmp
            )
            source = Dataset.from_dict(
                {"input": ["number"], "target": [1e-11], "alternative": [-3e-11]}
            )
            calls = []

            def preprocess(hf_dataset, dataset_manager):
                calls.append(dataset_manager.record)
                return hf_dataset

            preprocessor = TaskPreprocessor("numeric", preprocess, version="1")
            first = preprocessor(source, manager)
            second = preprocessor(source, manager, FieldSpec(target="alternative"))
            self.assertEqual(float(cast(str, first[0].target)), 1e-11)
            self.assertEqual(float(cast(str, second[0].target)), -3e-11)
            self.assertEqual(len(calls), 1)
            self.assertEqual(first.location, second.location)
            self.assertEqual(len(list((manager.version_path / "tasks").glob("*"))), 1)


if __name__ == "__main__":
    unittest.main()
