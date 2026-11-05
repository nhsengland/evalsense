"""Length-policy tests with synthetic token counts and a recording metric."""

import asyncio
from copy import deepcopy
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

MODULE = "evalsense.evaluation.evaluators.bertscore"


class RecordingMetric:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def compute(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(deepcopy(kwargs))
        return {
            "precision": [0.8],
            "recall": [0.6],
            "f1": [24 / 35],
            "hashcode": "fixture",
        }


def state(text: str) -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=1,
        epoch=1,
        input="input",
        messages=[ChatMessageUser(content="input")],
        output=ModelOutput.from_content(model="fixture", content=text),
    )


class LengthPolicyTests(unittest.TestCase):
    def setUp(self):
        self.backend = RecordingMetric()
        load = patch(f"{MODULE}.evaluate.load", return_value=self.backend)
        self.load = load.start()
        self.addCleanup(load.stop)
        context = patch(
            f"{MODULE}._load_length_context", return_value=(object(), 8), create=True
        )
        self.context = context.start()
        self.addCleanup(context.stop)
        encode = patch(
            f"{MODULE}._encoded_length",
            side_effect=lambda _, text: len(text.split()) + 2,
            create=True,
        )
        self.encode = encode.start()
        self.addCleanup(encode.stop)

    def test_prediction_and_reference_overflow_are_rejected_before_compute(self):
        for source in ("prediction", "reference"):
            with self.subTest(source=source):
                args: dict[str, Any] = dict(prediction="short", reference="short")
                args[source] = "word " * 7
                with self.assertRaisesRegex(
                    ValueError, f"{source} has 9 tokens.*limit is 8"
                ):
                    BertScoreCalculator().calculate(**args)
        self.assertEqual(self.backend.calls, [])

    def test_exact_limit_includes_special_tokens_and_checks_each_text_separately(self):
        text = "word " * 6
        result = BertScoreCalculator().calculate(prediction=text, reference=text)
        self.assertEqual(
            result.value,
            {
                "BERTScore Precision": 0.8,
                "BERTScore Recall": 0.6,
                "BERTScore F1": 24 / 35,
            },
        )
        self.assertEqual(result.metadata, {"hashcode": "fixture"})
        self.assertEqual(result.answer, text)
        self.assertEqual(self.backend.calls[0]["predictions"], [text])
        self.assertEqual(self.backend.calls[0]["references"], [text])

    def test_error_does_not_echo_patient_text(self):
        text = "sensitive patient information " * 4
        with self.assertRaises(ValueError) as error:
            BertScoreCalculator().calculate(prediction=text, reference="short")
        self.assertNotIn("sensitive", str(error.exception))
        self.assertIn("No score was computed", str(error.exception))

    def test_preflight_is_lazy_and_missing_reference_does_not_load_tokenizer(self):
        calculator = BertScoreCalculator()
        self.context.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Reference is required"):
            calculator.calculate(prediction="short")
        self.context.assert_not_called()
        self.assertEqual(self.backend.calls, [])

    def test_tokenizer_is_cached_per_model_and_fast_setting(self):
        calculator = BertScoreCalculator(model_type="model-one")
        for _ in range(2):
            calculator.calculate(prediction="short", reference="short")
        self.assertEqual(self.context.call_count, 1)
        calculator.calculate(
            prediction="short", reference="short", use_fast_tokenizer=True
        )
        self.context.assert_called_with("model-one", True)
        calculator.model_type = "model-two"
        calculator.calculate(
            prediction="short", reference="short", use_fast_tokenizer=True
        )
        self.context.assert_called_with("model-two", True)
        self.assertEqual(self.context.call_count, 3)

    def test_failure_is_not_cached_as_a_valid_result(self):
        calculator = BertScoreCalculator()
        with self.assertRaises(ValueError):
            calculator.calculate(prediction="word " * 7, reference="short")
        calculator.calculate(prediction="short", reference="short")
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.context.call_count, 1)

    def test_unknown_context_and_loader_failures_do_not_produce_scores(self):
        for failure in (
            ValueError("no finite limit"),
            OSError("missing configuration"),
        ):
            with self.subTest(failure=type(failure)):
                self.context.side_effect = failure
                with self.assertRaises(type(failure)):
                    BertScoreCalculator().calculate(
                        prediction="short", reference="short"
                    )
        self.assertEqual(self.backend.calls, [])

    def test_tokenization_errors_are_not_ignored(self):
        self.encode.side_effect = RuntimeError("tokenization failed")
        with self.assertRaisesRegex(RuntimeError, "tokenization failed"):
            BertScoreCalculator().calculate(prediction="short", reference="short")
        self.assertEqual(self.backend.calls, [])

    def test_embedding_batch_size_does_not_change_the_input_limit(self):
        calculator = BertScoreCalculator()
        for size in (1, 64, 4096):
            with self.subTest(batch_size=size), self.assertRaises(ValueError):
                calculator.calculate(
                    prediction="word " * 7, reference="short", batch_size=size
                )
        self.assertEqual(self.backend.calls, [])

    def test_other_compute_options_and_short_outputs_are_unchanged(self):
        options: dict[str, Any] = dict(
            verbose=True,
            device="cpu",
            batch_size=2,
            nthreads=3,
            rescale_with_baseline=True,
            baseline_path="baseline.tsv",
            use_fast_tokenizer=True,
        )
        calculator = BertScoreCalculator(
            model_type="fixture-model", lang="fr", num_layers=1
        )
        calculator.calculate(prediction="p", reference="r", **options)
        self.assertEqual(
            self.backend.calls,
            [
                dict(
                    predictions=["p"],
                    references=["r"],
                    model_type="fixture-model",
                    lang="fr",
                    num_layers=1,
                    idf=False,
                    **options,
                )
            ],
        )

    def test_async_and_sync_use_the_same_policy(self):
        calculator = BertScoreCalculator()
        synchronous = calculator.calculate(prediction="p", reference="r")
        asynchronous = asyncio.run(
            calculator.calculate_async(prediction="p", reference="r")
        )
        self.assertEqual(synchronous.model_dump(), asynchronous.model_dump())
        with self.assertRaises(ValueError):
            asyncio.run(
                calculator.calculate_async(prediction="p", reference="word " * 7)
            )
        self.assertEqual(len(self.backend.calls), 2)

    def test_concurrent_async_calls_do_not_disable_the_guard(self):
        calculator = BertScoreCalculator()

        async def run():
            return await asyncio.gather(
                calculator.calculate_async(prediction="short", reference="short"),
                calculator.calculate_async(prediction="word " * 7, reference="short"),
                calculator.calculate_async(prediction="short", reference="short"),
                return_exceptions=True,
            )

        results = asyncio.run(run())
        self.assertIsInstance(results[1], ValueError)
        self.assertEqual(len(self.backend.calls), 2)

    def test_factory_enforces_length_checks(self):
        evaluator = get_bertscore_evaluator(use_fast_tokenizer=True)
        scorer = cast(Scorer, evaluator.scorer)
        with self.assertRaises(ValueError):
            asyncio.run(scorer(state("word " * 7), Target("short")))
        self.assertEqual(self.backend.calls, [])
        self.context.assert_called_once_with("microsoft/deberta-xlarge-mnli", True)

    def test_cleanup_reload_validates_lengths_again(self):
        evaluator = get_bertscore_evaluator()
        scorer = cast(Scorer, evaluator.scorer)
        asyncio.run(scorer(state("short"), Target("short")))
        assert evaluator.cleanup_fun is not None
        with patch.dict(sys.modules, {"torch": Mock()}):
            evaluator.cleanup_fun()
        with self.assertRaises(ValueError):
            asyncio.run(scorer(state("word " * 7), Target("short")))
        self.assertEqual(self.context.call_count, 2)
        self.assertEqual(self.load.call_count, 2)
        self.assertEqual(len(self.backend.calls), 1)

    def test_declared_unbounded_context_preserves_complete_inputs(self):
        self.context.return_value = (object(), None)
        calculator = BertScoreCalculator(model_type="xlnet-base-cased")
        text = "word " * 600
        for result in (
            calculator.calculate(prediction=text, reference=text),
            asyncio.run(calculator.calculate_async(prediction=text, reference=text)),
        ):
            self.assertEqual(result.metadata, {"hashcode": "fixture"})
            self.assertEqual(result.answer, text)
        self.assertEqual(len(self.backend.calls), 2)
        for call in self.backend.calls:
            self.assertEqual(call["predictions"], [text])
            self.assertEqual(call["references"], [text])
        self.assertEqual(self.context.call_count, 1)

    def test_unbounded_context_does_not_disable_checks_after_model_change(self):
        self.context.side_effect = [(object(), None), (object(), 8)]
        calculator = BertScoreCalculator(model_type="xlnet-base-cased")
        calculator.calculate(prediction="word " * 9, reference="short")
        calculator.model_type = "finite-model"
        with self.assertRaisesRegex(ValueError, "limit is 8"):
            calculator.calculate(prediction="word " * 9, reference="short")
        self.assertEqual(len(self.backend.calls), 1)

    def test_public_error_type_can_be_caught_separately(self):
        from evalsense.evaluation.evaluators import BertScoreLengthError

        with self.assertRaises(BertScoreLengthError):
            BertScoreCalculator().calculate(prediction="word " * 7, reference="short")

    def test_pipeline_records_error_instead_of_a_truncated_success(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp) / "projects"),
        ):
            manager = HuggingFaceDatasetManager(
                "length-fixture", version="1", splits=["test"], data_dir=tmp
            )
            DatasetDict(
                {
                    "test": Dataset.from_dict(
                        {"id": ["sample"], "input": ["input"], "target": ["short"]}
                    )
                }
            ).save_to_disk(manager.main_data_path)

            @solver
            def fixed_generation() -> Solver:
                async def solve(state: TaskState, generate: Generate) -> TaskState:
                    state.output = ModelOutput.from_content(
                        model="fixture", content="word " * 7
                    )
                    return state

                return solve

            evaluator = get_bertscore_evaluator()
            evaluator.cleanup_fun = None
            args: dict[str, Any] = {}
            if "version" in inspect.signature(GenerationSteps).parameters:
                args["version"] = "1"
            experiment = ExperimentConfig(
                manager,
                GenerationSteps("fixture", fixed_generation(), **args),
                ModelConfig("mockllm/model"),
                evaluator=evaluator,
            )
            project = Project("length-check")
            Pipeline(experiment, project).run(
                show_progress=False,
                eval_kwargs={"display": "none", "log_format": "json"},
            )
            reopened = Project("length-check")
            generated = reopened.get_record(experiment.generation_record)
            evaluated = reopened.get_record(experiment.evaluation_record)
            assert generated is not None and evaluated is not None
            self.assertEqual(generated.status, "success")
            self.assertEqual(evaluated.status, "error")
            self.assertIn("9 tokens", evaluated.error_message or "")
            self.assertEqual(reopened.get_logs(type="evaluation", status="success"), {})
            self.assertEqual(self.backend.calls, [])


if __name__ == "__main__":
    unittest.main()
