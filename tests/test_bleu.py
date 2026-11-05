"""BLEU scorer, corpus aggregation and backend numerical regressions."""

import asyncio
import importlib.util
import math
import os
from pathlib import Path
import tempfile
from typing import Any, cast
import unittest
from unittest.mock import Mock, patch

import evaluate
from inspect_ai import score as score_log
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalSample,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import SampleScore, Score, Scorer, Target, mean
from inspect_ai.solver import TaskState

from evalsense.datasets import DatasetRecord
from evalsense.evaluation import EvaluationRecord, ResultRecord
from evalsense.evaluation.evaluators import (
    BleuPrecisionScoreCalculator,
    BleuScoreCalculator,
    bleu_metric,
    get_bleu_evaluator,
)
from evalsense.generation import ModelRecord
from evalsense.workflow import Project
from evalsense.workflow.analysers import TabularResultAnalyser

LOAD = "evalsense.evaluation.evaluators.bleu.evaluate.load"
BLEU_REVISION = "4bc9d73a0ebdfcb960e09da5166c86591fc219b0"


def state(prediction: str) -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=1,
        epoch=1,
        input="input",
        messages=[],
        output=ModelOutput.from_content(model="fixture", content=prediction),
    )


def source_log(pairs: list[tuple[str, str]]) -> EvalLog:
    return EvalLog(
        status="success",
        eval=EvalSpec(
            created="2026-01-01T00:00:00Z",
            task="fixture",
            model="mockllm/model",
            dataset=EvalDataset(name="fixture"),
            config=EvalConfig(),
        ),
        samples=[
            EvalSample(
                id=i,
                epoch=1,
                input="input",
                target=reference,
                output=ModelOutput.from_content(model="fixture", content=prediction),
            )
            for i, (prediction, reference) in enumerate(pairs)
        ],
    )


class BleuTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        # Distinct values make selecting unigram precision observable.
        self.backend.compute.return_value = {
            "bleu": 0.25,
            "precisions": [1.0, 0.5, 0.25, 0.125],
        }
        load = patch(LOAD, return_value=self.backend)
        self.load = load.start()
        self.addCleanup(load.stop)

    def test_sample_uses_bleu_and_records_its_definition(self):
        result = BleuScoreCalculator().calculate(
            prediction="prediction", reference="reference"
        )
        self.assertEqual(result.value, 0.25)
        self.assertEqual(result.answer, "prediction")
        self.assertEqual(
            result.metadata,
            {
                "prediction": "prediction",
                "reference": "reference",
                "bleu_sample_variant": "sentence_bleu_4",
                "max_order": 4,
                "smooth": False,
            },
        )
        self.backend.compute.assert_called_once_with(
            predictions=["prediction"],
            references=["reference"],
            max_order=4,
            smooth=False,
        )

    def test_legacy_precision_calculator_keeps_its_value_and_metadata(self):
        calculator = BleuPrecisionScoreCalculator()
        result = calculator.calculate(prediction="p", reference="r")
        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.metadata, {"prediction": "p", "reference": "r"})
        self.assertEqual(
            asyncio.run(
                calculator.calculate_async(prediction="p", reference="r")
            ).value,
            1.0,
        )

    def test_sentence_sync_and_async_agree(self):
        calculator = BleuScoreCalculator()
        a = calculator.calculate(prediction="p", reference="r")
        b = asyncio.run(calculator.calculate_async(prediction="p", reference="r"))
        self.assertEqual(a.model_dump(), b.model_dump())
        self.assertEqual(
            self.backend.compute.call_args_list[0],
            self.backend.compute.call_args_list[1],
        )

    def test_missing_or_blank_text_fails_before_compute(self):
        for prediction, reference in [
            ("p", None),
            ("p", ""),
            ("p", " \t"),
            ("", "r"),
            (" \n", "r"),
        ]:
            with (
                self.subTest(prediction=prediction, reference=reference),
                self.assertRaisesRegex(ValueError, "non-empty"),
            ):
                BleuScoreCalculator().calculate(
                    prediction=prediction, reference=reference
                )
        self.backend.compute.assert_not_called()

    def test_backend_errors_are_not_converted_to_scores(self):
        self.backend.compute.side_effect = RuntimeError("backend failure")
        with self.assertRaisesRegex(RuntimeError, "backend failure"):
            BleuScoreCalculator().calculate(prediction="p", reference="r")

    def test_factory_uses_sentence_values_with_distinct_corpus_label(self):
        evaluator = get_bleu_evaluator()
        result = score_log(source_log([("p", "r")]), cast(Scorer, evaluator.scorer))
        assert result.samples and result.samples[0].scores and result.results
        self.assertEqual(evaluator.name, "BLEU")
        self.assertEqual(list(result.samples[0].scores), ["Sentence BLEU"])
        self.assertEqual(result.samples[0].scores["Sentence BLEU"].value, 0.25)
        self.assertEqual(set(result.results.scores[0].metrics), {"BLEU (corpus)"})

    def test_custom_names_and_metrics_are_preserved(self):
        evaluator = get_bleu_evaluator("study", "sentence overlap", [mean()])
        result = score_log(source_log([("p", "r")]), cast(Scorer, evaluator.scorer))
        assert result.results and result.samples and result.samples[0].scores
        self.assertEqual(evaluator.name, "study")
        self.assertEqual(list(result.samples[0].scores), ["sentence overlap"])
        self.assertEqual(set(result.results.scores[0].metrics), {"mean"})
        self.assertEqual(result.results.scores[0].metrics["mean"].value, 0.25)

    def test_corpus_recomputes_from_text_not_sample_values(self):
        samples = [
            SampleScore(
                score=Score(value=99.0, metadata={"prediction": p, "reference": r})
            )
            for p, r in [("p1", "r1"), ("p2", "r2")]
        ]
        self.assertEqual(bleu_metric()(samples), 0.25)
        self.backend.compute.assert_called_once_with(
            predictions=["p1", "p2"],
            references=["r1", "r2"],
            max_order=4,
            smooth=False,
        )

    def test_scored_log_roundtrip_and_summary_keep_both_levels(self):
        evaluator = get_bleu_evaluator()
        scored = score_log(source_log([("p", "r")]), cast(Scorer, evaluator.scorer))
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp)),
        ):
            project = Project("bleu")
            project.evaluation_log_path.mkdir(parents=True)
            path = project.evaluation_log_path / "2026_task_bleu.json"
            write_eval_log(scored, location=str(path))
            key = EvaluationRecord(
                dataset_record=DatasetRecord(
                    name="fixture", version="1", splits=("test",)
                ),
                model_record=ModelRecord(name="mockllm/model"),
                task_name="task",
                generator_name="generator",
                evaluator_name=evaluator.name,
            )
            project.update_record(
                key, ResultRecord(status="success", log_location=str(path))
            )
            reopened = Project("bleu")
            stored = reopened.get_log(key)
            assert stored and stored.samples and stored.samples[0].scores
            self.assertEqual(stored.samples[0].scores["Sentence BLEU"].value, 0.25)
            table = TabularResultAnalyser()(reopened)
            self.assertIn("Sentence BLEU/BLEU (corpus)", table.columns)

    @unittest.skipUnless(
        importlib.util.find_spec("gradio"), "webui extra not installed"
    )
    def test_ui_defaults_follow_the_scorer_contract(self):
        from evalsense.webui.configurators.evaluators.bleu import BleuConfigurator

        configurator = BleuConfigurator()
        options = {
            entry["input_name"]: cast(Any, entry["component"]).value
            for entry in configurator.input_widget()
        }
        self.assertEqual(options["scorer_name"], "Sentence BLEU")
        scorer = cast(Scorer, configurator.instantiate_evaluator(**options).scorer)
        result = asyncio.run(scorer(state("p"), Target("r")))
        assert result is not None
        self.assertEqual(result.value, 0.25)


@unittest.skipUnless(
    os.environ.get("EVALSENSE_TEST_NATIVE_BLEU") == "1",
    "enable the reference backend tests",
)
class NativeBleuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Pin the test oracle independently of the application's metric loading.
        # Only metric source is downloaded; there are no model calls or datasets.
        cls.backend = evaluate.load("bleu", revision=BLEU_REVISION)

    def setUp(self):
        load = patch(LOAD, return_value=self.backend)
        load.start()
        self.addCleanup(load.stop)

    def test_identical_and_reordered_words_have_different_sentence_bleu(self):
        reference = "one two three four"
        calculator = BleuScoreCalculator()
        self.assertEqual(
            calculator.calculate(prediction=reference, reference=reference).value, 1.0
        )
        reversed_text = "four three two one"
        self.assertEqual(
            calculator.calculate(prediction=reversed_text, reference=reference).value,
            0.0,
        )
        self.assertEqual(
            BleuPrecisionScoreCalculator()
            .calculate(prediction=reversed_text, reference=reference)
            .value,
            1.0,
        )

    def test_brevity_penalty_matches_hand_calculation(self):
        reference = "one two three four five six seven eight"
        prediction = "one two three four"
        result = BleuScoreCalculator().calculate(
            prediction=prediction, reference=reference
        )
        self.assertAlmostEqual(cast(float, result.value), math.exp(-1.0), places=12)
        self.assertEqual(
            BleuPrecisionScoreCalculator()
            .calculate(prediction=prediction, reference=reference)
            .value,
            1.0,
        )

    def test_ngram_clipping_and_geometric_mean_match_hand_calculation(self):
        # Matched counts: 5/6, 4/5, 3/4, 2/3; equal lengths, so BP=1.
        result = BleuScoreCalculator().calculate(
            prediction="one two three four five five",
            reference="one two three four five six",
        )
        self.assertAlmostEqual(
            cast(float, result.value),
            ((5 / 6) * (4 / 5) * (3 / 4) * (2 / 3)) ** 0.25,
            places=12,
        )

    def test_unsmoothed_short_and_zero_overlap_values_are_zero(self):
        for prediction, reference in [
            ("one", "one"),
            ("one two three", "one two three"),
            ("x y z w", "one two three four"),
        ]:
            with self.subTest(prediction=prediction):
                self.assertEqual(
                    BleuScoreCalculator()
                    .calculate(prediction=prediction, reference=reference)
                    .value,
                    0.0,
                )

    def test_corpus_is_not_mean_sentence_bleu(self):
        good = "one two three four five six seven eight nine ten"
        pairs = [(good, good), ("x y z w", "one two three four")]
        calculator = BleuScoreCalculator()
        scores = [calculator.calculate(prediction=p, reference=r) for p, r in pairs]
        sentence_mean = sum(cast(float, s.value) for s in scores) / 2
        corpus = bleu_metric()([SampleScore(score=s) for s in scores])
        self.assertEqual(sentence_mean, 0.5)
        self.assertAlmostEqual(
            cast(float, corpus),
            ((10 / 14) * (9 / 12) * (8 / 10) * (7 / 8)) ** 0.25,
            places=12,
        )
        self.assertNotEqual(corpus, sentence_mean)

    def test_actual_inspect_path_and_corpus_metric(self):
        reference = "one two three four"
        evaluator = get_bleu_evaluator()
        scored = score_log(
            source_log([(reference, reference), ("four three two one", reference)]),
            cast(Scorer, evaluator.scorer),
        )
        assert scored.samples and scored.results
        values = [s.scores["Sentence BLEU"].value for s in scored.samples if s.scores]
        self.assertEqual(values, [1.0, 0.0])
        corpus = scored.results.scores[0].metrics["BLEU (corpus)"].value
        self.assertAlmostEqual(corpus, (0.5**3) ** 0.25, places=12)


if __name__ == "__main__":
    unittest.main()
