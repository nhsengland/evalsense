"""Cache reuse boundaries, using synthetic data and no model downloads."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Any, cast
import unittest
from unittest.mock import AsyncMock, Mock, patch

from datasets import Dataset
from inspect_ai.scorer import Score, Scorer, Target
from inspect_ai.solver import TaskState

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation.evaluators import get_bertscore_evaluator
from evalsense.tasks import TaskPreprocessor
from evalsense.utils.cache import CacheIdentityError, fingerprint


class CachePolicyTests(unittest.TestCase):
    def test_bertscore_ignores_logging_and_unweighted_idf_threads(self):
        with patch("evalsense.evaluation.evaluators.bertscore.evaluate.load"):
            baseline = get_bertscore_evaluator().cache_key
            for options in (
                {"verbose": True},
                {"nthreads": 4},
                {"verbose": True, "nthreads": 8},
            ):
                with self.subTest(options=options):
                    other = get_bertscore_evaluator(**cast(dict[str, Any], options))
                    self.assertEqual(baseline, other.cache_key)

    def test_bertscore_keeps_device_and_batch_size_in_identity(self):
        with patch("evalsense.evaluation.evaluators.bertscore.evaluate.load"):
            baseline = get_bertscore_evaluator(device="cpu", batch_size=8).cache_key
            for options in (
                {"device": "cuda", "batch_size": 8},
                {"device": "cpu", "batch_size": 1},
            ):
                with self.subTest(options=options):
                    other = get_bertscore_evaluator(**cast(dict[str, Any], options))
                    self.assertNotEqual(baseline, other.cache_key)

    def test_excluded_options_are_still_passed_to_the_calculator(self):
        with (
            patch("evalsense.evaluation.evaluators.bertscore.evaluate.load"),
            patch(
                "evalsense.evaluation.evaluators.bertscore.BertScoreCalculator.calculate_async",
                new_callable=AsyncMock,
                return_value=Score(value=1),
            ) as calculate,
        ):
            evaluator = get_bertscore_evaluator(verbose=True, nthreads=4)
            state = cast(
                TaskState, SimpleNamespace(output=SimpleNamespace(completion="answer"))
            )
            asyncio.run(cast(Scorer, evaluator.scorer)(state, Target("reference")))
        self.assertEqual(calculate.await_count, 1)
        assert calculate.await_args is not None
        self.assertIs(calculate.await_args.kwargs["verbose"], True)
        self.assertEqual(calculate.await_args.kwargs["nthreads"], 4)

    def test_exact_package_versions_remain_in_the_fingerprint(self):
        versions = {"evalsense": "0.1.6", "inspect-ai": "0.3.205"}
        with patch(
            "evalsense.utils.cache._implementation_versions", return_value=versions
        ):
            baseline = fingerprint({"prompt": "same"})
            self.assertEqual(baseline, fingerprint({"prompt": "same"}))
        for package, version in (("evalsense", "0.1.7"), ("inspect-ai", "0.3.206")):
            with (
                self.subTest(package=package),
                patch(
                    "evalsense.utils.cache._implementation_versions",
                    return_value=versions | {package: version},
                ),
            ):
                self.assertNotEqual(baseline, fingerprint({"prompt": "same"}))

    def test_invalid_dataset_fingerprints_fail_before_cache_access(self):
        missing = object()
        for value in (missing, None, "", " \t", 0, False, [], {}):
            for force in (False, True):
                with (
                    self.subTest(value=repr(value), force=force),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    manager = HuggingFaceDatasetManager(
                        "fixture", version="1", splits=["test"], data_dir=tmp
                    )
                    data = Dataset.from_dict({"input": ["input"], "target": ["yes"]})
                    if value is missing:
                        delattr(data, "_fingerprint")
                    else:
                        setattr(data, "_fingerprint", value)
                    preprocessing = Mock()
                    pre = TaskPreprocessor("fixture", preprocessing, version="1")
                    with patch(
                        "evalsense.tasks.task_preprocessor.json_dataset"
                    ) as load:
                        with self.assertRaisesRegex(
                            CacheIdentityError, "fingerprint.*missing or invalid"
                        ):
                            pre(data, manager, force_reprocess=force)
                        load.assert_not_called()
                    preprocessing.assert_not_called()
                    self.assertFalse(list(Path(tmp).rglob("*.jsonl")))

    def test_existing_dataset_fingerprint_keeps_the_same_cache_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="1", splits=["test"], data_dir=tmp
            )
            data = Dataset.from_dict({"input": ["input"], "target": ["yes"]})
            calls = []

            def preprocessing(hf_dataset, dataset_manager):
                calls.append(hf_dataset)
                return hf_dataset

            pre = TaskPreprocessor("fixture", preprocessing, version="1")
            key = fingerprint(
                {
                    "dataset": manager.record,
                    "content": data._fingerprint,
                    "preprocessor": pre.cache_key,
                }
            )
            expected = manager.version_path / "tasks" / f"fixture-{key}.jsonl"
            self.assertEqual(pre(data, manager)[0].input, "input")
            self.assertTrue(expected.exists())
            self.assertEqual(pre(data, manager)[0].input, "input")
            self.assertEqual(len(calls), 1)
            self.assertEqual(list(expected.parent.glob("*.jsonl")), [expected])


if __name__ == "__main__":
    unittest.main()
