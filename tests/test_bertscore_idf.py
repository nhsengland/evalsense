"""Pairwise BERTScore guards and wrapper behavior, without downloading a model."""

import asyncio
from collections import defaultdict
from copy import deepcopy
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
from typing import Any, cast
import unittest
from unittest.mock import Mock, patch

from datasets import Dataset, DatasetDict
from inspect_ai.model import ChatMessageUser, ModelName, ModelOutput
from inspect_ai.scorer import Scorer, Target
from inspect_ai.solver import Generate, Solver, TaskState, solver

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import ExperimentConfig
from evalsense.evaluation.evaluators import BertScoreCalculator, get_bertscore_evaluator
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.workflow import Pipeline, Project


LOAD_METRIC = "evalsense.evaluation.evaluators.bertscore.evaluate.load"
VALUES = {
    "BERTScore Precision": 0.9,
    "BERTScore Recall": 0.6,
    "BERTScore F1": 0.72,
}


class RecordingMetric:
    """Return controlled values and record the actual inputs to the backend."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def compute(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(deepcopy(kwargs))
        return {
            "precision": [0.9],
            "recall": [0.6],
            "f1": [0.72],
            "hashcode": "fixture-no-idf",
        }


def task_state(prediction: str = "prediction") -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id="sample",
        epoch=1,
        input="input",
        messages=[ChatMessageUser(content="input")],
        output=ModelOutput.from_content(model="mockllm/model", content=prediction),
    )


class PairwiseIdfTests(unittest.TestCase):
    def setUp(self):
        # This recording backend has no tokenizer; length checks have their own suite.
        length_check = patch.object(
            BertScoreCalculator, "_validate_input_lengths", create=True
        )
        length_check.start()
        self.addCleanup(length_check.stop)

    def test_automatic_idf_is_rejected_before_metric_loading(self):
        with patch(LOAD_METRIC) as load:
            with self.assertRaisesRegex(ValueError, "reference corpus"):
                BertScoreCalculator(idf=True)
            load.assert_not_called()

    def test_factory_rejects_automatic_idf_before_metric_loading(self):
        with patch(LOAD_METRIC) as load:
            with self.assertRaisesRegex(ValueError, "idf=False"):
                get_bertscore_evaluator(idf=True)
            load.assert_not_called()

    def test_idf_maps_are_not_treated_as_a_boolean_flag(self):
        mappings = [
            {101: 0.0, 102: 0.0, 2001: 1.5},
            {"clinical": 1.5},
            defaultdict(lambda: 1.0, {101: 0.0}),
            {},
        ]
        for mapping in mappings:
            for constructor in (BertScoreCalculator, get_bertscore_evaluator):
                with self.subTest(
                    mapping=type(mapping), constructor=constructor.__name__
                ):
                    with patch(LOAD_METRIC) as load:
                        with self.assertRaisesRegex(
                            ValueError, "precomputed IDF dictionaries"
                        ):
                            constructor(idf=cast(Any, mapping))
                        load.assert_not_called()

    def test_only_the_boolean_false_is_accepted(self):
        for value in (None, 0, 1, 0.0, "false", "true", [], [1]):
            with self.subTest(value=value), patch(LOAD_METRIC) as load:
                with self.assertRaisesRegex(ValueError, "idf=False"):
                    BertScoreCalculator(idf=cast(Any, value))
                load.assert_not_called()

    def test_defaults_match_explicit_unweighted_scoring(self):
        backend = RecordingMetric()
        with patch(LOAD_METRIC, return_value=backend):
            default = BertScoreCalculator()
            explicit = BertScoreCalculator(idf=False)
        a = default.calculate(prediction="prediction", reference="reference")
        b = explicit.calculate(prediction="prediction", reference="reference")
        self.assertEqual(a.value, VALUES)
        self.assertEqual(a.model_dump(), b.model_dump())
        self.assertEqual(a.metadata, {"hashcode": "fixture-no-idf"})
        self.assertEqual(a.answer, "prediction")
        self.assertEqual(backend.calls[0], backend.calls[1])
        self.assertIs(backend.calls[0]["idf"], False)
        self.assertEqual(backend.calls[0]["predictions"], ["prediction"])
        self.assertEqual(backend.calls[0]["references"], ["reference"])

    def test_other_metric_parameters_are_preserved(self):
        backend = RecordingMetric()
        with patch(LOAD_METRIC, return_value=backend):
            calculator = BertScoreCalculator(
                model_type="fixture-model", lang="fr", num_layers=4
            )
        options: dict[str, Any] = dict(
            verbose=True,
            device="cpu",
            batch_size=8,
            nthreads=2,
            rescale_with_baseline=True,
            baseline_path="fixture.tsv",
            use_fast_tokenizer=True,
        )
        calculator.calculate(prediction="p", reference="r", **options)
        self.assertEqual(
            backend.calls,
            [
                dict(
                    predictions=["p"],
                    references=["r"],
                    model_type="fixture-model",
                    lang="fr",
                    num_layers=4,
                    idf=False,
                    **options,
                )
            ],
        )

    def test_batch_size_does_not_enable_single_reference_idf(self):
        with patch(LOAD_METRIC) as load:
            for batch_size in (1, 64, 1024):
                with self.subTest(batch_size=batch_size):
                    with self.assertRaisesRegex(ValueError, "one pair at a time"):
                        get_bertscore_evaluator(idf=True, batch_size=batch_size)
            load.assert_not_called()

    def test_mutated_idf_is_rejected_before_backend_compute(self):
        for value in (True, {}, {42: 1.0}):
            backend = RecordingMetric()
            with self.subTest(value=value), patch(LOAD_METRIC, return_value=backend):
                calculator = BertScoreCalculator()
                calculator.idf = cast(Any, value)
                with self.assertRaisesRegex(ValueError, "idf=False"):
                    calculator.calculate(prediction="p", reference="r")
                self.assertEqual(backend.calls, [])

    def test_idf_requests_in_call_kwargs_are_not_silently_ignored(self):
        backend = RecordingMetric()
        with patch(LOAD_METRIC, return_value=backend):
            calculator = BertScoreCalculator()
        for value in (True, {}, {42: 1.0}):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "idf=False"),
            ):
                calculator.calculate(prediction="p", reference="r", idf=value)
        self.assertEqual(backend.calls, [])
        calculator.calculate(prediction="p", reference="r", idf=False)
        self.assertIs(backend.calls[0]["idf"], False)

    def test_backend_errors_are_not_converted_to_scores(self):
        backend = Mock()
        backend.compute.side_effect = RuntimeError("backend failed")
        with patch(LOAD_METRIC, return_value=backend):
            calculator = BertScoreCalculator()
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            calculator.calculate(prediction="p", reference="r")

    def test_missing_reference_still_raises_without_computing(self):
        backend = RecordingMetric()
        with patch(LOAD_METRIC, return_value=backend):
            calculator = BertScoreCalculator()
        with self.assertRaisesRegex(ValueError, "Reference is required"):
            calculator.calculate(prediction="p")
        self.assertEqual(backend.calls, [])

    @unittest.skipUnless(
        importlib.util.find_spec("gradio"), "webui extra not installed"
    )
    def test_ui_configuration_cannot_bypass_validation(self):
        from evalsense.webui.configurators.evaluators.bertscore import (
            BertScoreConfigurator,
        )

        with patch(LOAD_METRIC) as load:
            with self.assertRaisesRegex(ValueError, "reference corpus"):
                BertScoreConfigurator().instantiate_evaluator(idf=True)
            load.assert_not_called()


class AsyncIdfTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # This recording backend has no tokenizer; length checks have their own suite.
        length_check = patch.object(
            BertScoreCalculator, "_validate_input_lengths", create=True
        )
        length_check.start()
        self.addCleanup(length_check.stop)

    async def test_async_and_sync_scores_are_identical(self):
        backend = RecordingMetric()
        with patch(LOAD_METRIC, return_value=backend):
            calculator = BertScoreCalculator()
        options: dict[str, Any] = dict(
            prediction="p",
            reference="r",
            device="cpu",
            batch_size=4,
            rescale_with_baseline=True,
            baseline_path="baseline.tsv",
        )
        synchronous = calculator.calculate(**options)
        asynchronous = await calculator.calculate_async(**options)
        self.assertEqual(synchronous.model_dump(), asynchronous.model_dump())
        self.assertEqual(backend.calls[0], backend.calls[1])

    async def test_async_guard_catches_mutated_configuration(self):
        backend = RecordingMetric()
        with patch(LOAD_METRIC, return_value=backend):
            calculator = BertScoreCalculator()
        calculator.idf = True
        with self.assertRaisesRegex(ValueError, "reference corpus"):
            await calculator.calculate_async(prediction="p", reference="r")
        self.assertEqual(backend.calls, [])

    async def test_async_idf_call_kwargs_are_not_silently_ignored(self):
        backend = RecordingMetric()
        with patch(LOAD_METRIC, return_value=backend):
            calculator = BertScoreCalculator()
        for value in (True, {}, {42: 1.0}):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "idf=False"),
            ):
                await calculator.calculate_async(
                    prediction="p", reference="r", idf=value
                )
        self.assertEqual(backend.calls, [])
        await calculator.calculate_async(prediction="p", reference="r", idf=False)
        self.assertIs(backend.calls[0]["idf"], False)

    async def test_repeated_and_concurrent_calls_remain_unweighted(self):
        backend = RecordingMetric()
        with patch(LOAD_METRIC, return_value=backend):
            calculator = BertScoreCalculator()
        references = ["alpha beta", "beta gamma", "alpha beta"]
        for order in (references, list(reversed(references))):
            results = await asyncio.gather(
                *(
                    calculator.calculate_async(prediction="p", reference=reference)
                    for reference in order
                )
            )
            self.assertEqual([score.value for score in results], [VALUES] * len(order))
        self.assertEqual(len(backend.calls), 6)
        self.assertTrue(all(call["idf"] is False for call in backend.calls))
        self.assertTrue(all(len(call["references"]) == 1 for call in backend.calls))

    async def test_factory_passes_settings_to_its_scorer(self):
        backend = RecordingMetric()
        with patch(LOAD_METRIC, return_value=backend):
            evaluator = get_bertscore_evaluator(
                name="fixture", device="cpu", batch_size=2
            )
            result = await cast(Scorer, evaluator.scorer)(
                task_state(), Target("reference")
            )
        assert result is not None
        self.assertEqual(evaluator.name, "fixture")
        self.assertEqual(result.value, VALUES)
        self.assertIs(backend.calls[0]["idf"], False)
        self.assertEqual(backend.calls[0]["device"], "cpu")
        self.assertEqual(backend.calls[0]["batch_size"], 2)
        self.assertEqual(backend.calls[0]["references"], ["reference"])

    async def test_reloaded_factory_backend_stays_unweighted(self):
        first, second = RecordingMetric(), RecordingMetric()
        with patch(LOAD_METRIC, side_effect=[first, second]) as load:
            evaluator = get_bertscore_evaluator()
            before = await cast(Scorer, evaluator.scorer)(
                task_state(), Target("reference")
            )
            assert evaluator.cleanup_fun is not None
            # The fixture backend has no tensor resources. Keep the cleanup import
            # optional so these regression tests also run with base dependencies.
            with patch.dict(sys.modules, {"torch": Mock()}):
                evaluator.cleanup_fun()
            after = await cast(Scorer, evaluator.scorer)(
                task_state(), Target("reference")
            )
        assert before is not None and after is not None
        self.assertEqual(load.call_count, 2)
        self.assertEqual(before.model_dump(), after.model_dump())
        self.assertIs(first.calls[0]["idf"], False)
        self.assertIs(second.calls[0]["idf"], False)


class PipelineIdfTests(unittest.TestCase):
    def setUp(self):
        # This recording backend has no tokenizer; length checks have their own suite.
        length_check = patch.object(
            BertScoreCalculator, "_validate_input_lengths", create=True
        )
        length_check.start()
        self.addCleanup(length_check.stop)

    def test_unweighted_scores_survive_pipeline_and_project_reload(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp) / "projects"),
        ):
            manager = HuggingFaceDatasetManager(
                "bertscore-fixture", version="1", splits=["test"], data_dir=tmp
            )
            DatasetDict(
                {
                    "test": Dataset.from_dict(
                        {"id": ["sample"], "input": ["input"], "target": ["reference"]}
                    )
                }
            ).save_to_disk(manager.main_data_path)

            @solver
            def fixed_generation() -> Solver:
                async def solve(state: TaskState, generate: Generate) -> TaskState:
                    state.output = ModelOutput.from_content(
                        model="fixture", content="prediction"
                    )
                    return state

                return solve

            backend = RecordingMetric()
            with patch(LOAD_METRIC, return_value=backend):
                evaluator = get_bertscore_evaluator(device="cpu")
            # There is no transformer to release in the recording backend.
            evaluator.cleanup_fun = None
            steps_kwargs: dict[str, Any] = {}
            if "version" in inspect.signature(GenerationSteps).parameters:
                steps_kwargs["version"] = "1"
            experiment = ExperimentConfig(
                manager,
                GenerationSteps("fixture", fixed_generation(), **steps_kwargs),
                ModelConfig("mockllm/model"),
                evaluator=evaluator,
            )
            project = Project("idf-regression")
            Pipeline(experiment, project).run(
                show_progress=False,
                eval_kwargs={"display": "none", "log_format": "json"},
            )
            result = project.get_record(experiment.evaluation_record)
            assert result is not None
            self.assertEqual(result.status, "success")
            reopened = Project("idf-regression")
            log = reopened.get_log(experiment.evaluation_record)
            assert log is not None and log.samples and log.samples[0].scores
            score = next(iter(log.samples[0].scores.values()))
            self.assertEqual(score.value, VALUES)
            self.assertEqual(score.metadata, {"hashcode": "fixture-no-idf"})
            self.assertEqual(len(backend.calls), 1)
            self.assertIs(backend.calls[0]["idf"], False)


if __name__ == "__main__":
    unittest.main()
