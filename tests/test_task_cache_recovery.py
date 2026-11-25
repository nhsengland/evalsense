"""Task cache writes retain complete data when serialisation is interrupted."""

import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
from inspect_ai.dataset import Sample

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.tasks import DefaultTaskPreprocessor, TaskPreprocessor


def task_files(manager):
    return [
        path
        for path in (manager.version_path / "tasks").iterdir()
        if not path.name.startswith("._")
    ]


def numeric_target(sample: Sample) -> float:
    assert isinstance(sample.target, str)
    return float(sample.target)


class TaskCacheRecoveryTests(unittest.TestCase):
    def test_failed_first_write_leaves_no_cache_and_retry_keeps_all_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="1", splits=["test"], data_dir=tmp
            )
            source = Dataset.from_dict(
                {"input": ["first", "last"], "target": [1e-11, 2e-11]}
            )
            invalid = source.add_column("confidence", [0.5, float("nan")])
            outputs = iter([invalid, source])
            calls = []

            def preprocess(hf_dataset, dataset_manager):
                calls.append(hf_dataset)
                return next(outputs)

            pre = TaskPreprocessor("fixture", preprocess, version="1")
            with self.assertRaises(ValueError):
                pre(source, manager)
            self.assertEqual(task_files(manager), [])

            result = pre(source, manager)
            self.assertEqual([sample.input for sample in result], ["first", "last"])
            self.assertEqual(
                [numeric_target(sample) for sample in result], [1e-11, 2e-11]
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(task_files(manager)), 1)

    def test_failed_refresh_preserves_previous_cache_until_successful_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="1", splits=["test"], data_dir=tmp
            )
            source = Dataset.from_dict(
                {"input": ["first", "last"], "target": [1e-11, 2e-11]}
            )
            invalid = source.add_column("confidence", [0.5, float("inf")])
            refreshed = Dataset.from_dict(
                {"input": ["new first", "new last"], "target": [-1e-11, -2e-11]}
            )
            outputs = iter([source, invalid, refreshed])
            calls = []

            def preprocess(hf_dataset, dataset_manager):
                calls.append(hf_dataset)
                return next(outputs)

            pre = TaskPreprocessor("fixture", preprocess, version="1")
            initial = pre(source, manager)
            cache = task_files(manager)[0]
            original_bytes = cache.read_bytes()
            with self.assertRaises(ValueError):
                pre(source, manager, force_reprocess=True)
            self.assertEqual(task_files(manager), [cache])
            self.assertEqual(cache.read_bytes(), original_bytes)
            cached = pre(source, manager)
            self.assertEqual([s.input for s in cached], [s.input for s in initial])
            self.assertEqual(len(calls), 2)

            result = pre(source, manager, force_reprocess=True)
            self.assertEqual([s.input for s in result], ["new first", "new last"])
            self.assertEqual([numeric_target(s) for s in result], [-1e-11, -2e-11])
            self.assertNotEqual(cache.read_bytes(), original_bytes)
            self.assertEqual(task_files(manager), [cache])
            self.assertEqual(len(calls), 3)

    def test_keyboard_interrupt_after_a_batch_does_not_publish_a_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="1", splits=["test"], data_dir=tmp
            )
            source = Dataset.from_dict(
                {"input": [str(i) for i in range(1001)], "target": [1e-11] * 1001}
            )
            original_iter = Dataset.iter

            def interrupted(data, *args, **kwargs):
                batches = original_iter(data, *args, **kwargs)
                yield next(batches)
                raise KeyboardInterrupt("interrupted fixture")

            pre = DefaultTaskPreprocessor()
            with patch.object(Dataset, "iter", interrupted):
                with self.assertRaisesRegex(KeyboardInterrupt, "interrupted fixture"):
                    pre(source, manager)
            self.assertEqual(task_files(manager), [])

            result = pre(source, manager)
            self.assertEqual(len(result), 1001)
            self.assertEqual(result[-1].input, "1000")
            self.assertEqual(numeric_target(result[-1]), 1e-11)
            self.assertEqual(len(task_files(manager)), 1)

    def test_failed_atomic_replacement_keeps_the_previous_cache_state(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as tmp:
                manager = HuggingFaceDatasetManager(
                    "fixture", version="1", splits=["test"], data_dir=tmp
                )
                source = Dataset.from_dict(
                    {"input": ["first", "last"], "target": [1e-11, -2e-11]}
                )
                pre = DefaultTaskPreprocessor()
                if existing:
                    pre(source, manager)
                before = {
                    path: path.read_bytes()
                    for path in (manager.version_path / "tasks").glob("*.jsonl")
                }
                with patch(
                    "evalsense.tasks._transport.Path.replace",
                    side_effect=OSError("fixture storage failure"),
                ) as replace:
                    with self.assertRaisesRegex(OSError, "fixture storage failure"):
                        pre(source, manager, force_reprocess=True)
                replace.assert_called_once()
                self.assertEqual(
                    {path: path.read_bytes() for path in task_files(manager)}, before
                )
                result = pre(source, manager, force_reprocess=True)
                self.assertEqual([numeric_target(s) for s in result], [1e-11, -2e-11])
                self.assertEqual(len(task_files(manager)), 1)


if __name__ == "__main__":
    unittest.main()
