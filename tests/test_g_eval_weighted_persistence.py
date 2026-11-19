"""Retain weighted score evidence and failures across saved project lifecycles."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
import unittest
from unittest.mock import patch

from inspect_ai.log import EvalMetric, EvalResults, EvalScore, write_eval_log
from inspect_ai.model import Model, ModelOutput
from inspect_ai.scorer import Score

from evalsense.evaluation import ResultRecord
from evalsense.evaluation.evaluators.g_eval import (
    ESTIMATOR_FORMAT,
    GEvalScorerFactory,
    WEIGHTING_POLICY,
    get_g_eval_evaluator,
)
from evalsense.generation import ModelConfig
from evalsense.workflow import Pipeline, Project
from test_g_eval_persistence import fixture
from test_g_eval_weighted_ratings import rating_options, weighted_output
from test_g_eval_weighted_scoring import split_rating_output


def weighted_fixture(tmp: str, outputs: list[ModelOutput], *, logprobs: bool = True):
    experiment, project, judge = fixture(tmp, [])
    judge.generate.side_effect = outputs
    evaluator = get_g_eval_evaluator(
        prompt_template="Rate {prediction}",
        model_config=ModelConfig("mockllm/judge"),
        logprobs=logprobs,
    )
    evaluator.scorer = cast(GEvalScorerFactory, evaluator.scorer).create_scorer(
        cast(Model, judge)
    )
    evaluator.model_config = None
    return replace(experiment, evaluator=evaluator), project, judge


def complete_rating() -> ModelOutput:
    return weighted_output("9", "9", rating_options({9: 0.3, 10: 0.7}))


class WeightedPersistenceTests(unittest.TestCase):
    def test_score_estimator_and_probability_evidence_survive_reopening(self):
        for suffix in ("json", "eval"):
            for use_weights in (True, False):
                with (
                    self.subTest(suffix=suffix, weighted=use_weights),
                    TemporaryDirectory() as tmp,
                    patch(
                        "evalsense.workflow.project.PROJECTS_PATH",
                        Path(tmp) / "projects",
                    ),
                ):
                    output = complete_rating() if use_weights else split_rating_output()
                    experiment, project, _ = weighted_fixture(
                        tmp, [output], logprobs=use_weights
                    )
                    Pipeline(experiment, project).run(
                        show_progress=False,
                        eval_kwargs={"display": "none", "log_format": suffix},
                    )
                    reopened = Project("run")
                    record = reopened.get_record(experiment.evaluation_record)
                    assert record and record.log_location
                    self.assertEqual(record.status, "success")
                    self.assertEqual(Path(record.log_location).suffix, "." + suffix)
                    result = reopened.get_log(experiment.evaluation_record)
                    assert result and result.samples and result.samples[0].scores
                    rating = next(iter(result.samples[0].scores.values()))
                    assert rating.metadata
                    self.assertEqual(
                        rating.metadata["estimator_format"], ESTIMATOR_FORMAT
                    )
                    self.assertEqual(
                        rating.metadata["estimator"],
                        "weighted_rating_tokens" if use_weights else "sampled_rating",
                    )
                    self.assertAlmostEqual(
                        cast(float, rating.value), 8.7 / 9 if use_weights else 1.0
                    )
                    self.assertAlmostEqual(
                        rating.metadata["raw_score"], 9.7 if use_weights else 10
                    )
                    if use_weights:
                        self.assertEqual(
                            rating.metadata["weighting_policy"], WEIGHTING_POLICY
                        )
                        probabilities = rating.metadata["rating_probabilities"]
                        self.assertEqual(
                            set(probabilities), {str(i) for i in range(1, 11)}
                        )
                        self.assertAlmostEqual(probabilities["9"], 0.3)
                        self.assertAlmostEqual(probabilities["10"], 0.7)
                        self.assertAlmostEqual(
                            rating.metadata["rating_probability_mass"], 1.0
                        )
                    else:
                        self.assertNotIn("weighting_policy", rating.metadata)
                        self.assertNotIn("rating_probabilities", rating.metadata)
                        self.assertNotIn("rating_probability_mass", rating.metadata)

    def test_weighting_failure_survives_reopening_and_retry_preserves_generation(self):
        for suffix in ("json", "eval"):
            with (
                self.subTest(suffix=suffix),
                TemporaryDirectory() as tmp,
                patch(
                    "evalsense.workflow.project.PROJECTS_PATH", Path(tmp) / "projects"
                ),
            ):
                experiment, project, judge = weighted_fixture(
                    tmp,
                    [
                        weighted_output("9", "9", [("9", 0.3), ("1", 0.7)]),
                        complete_rating(),
                    ],
                )
                Pipeline(experiment, project).run(
                    show_progress=False,
                    eval_kwargs={"display": "none", "log_format": suffix},
                )
                reopened = Project("run")
                failed = reopened.get_record(experiment.evaluation_record)
                assert failed
                self.assertEqual(failed.status, "error")
                self.assertIn(
                    "complete single-token alternative", failed.error_message or ""
                )
                self.assertEqual(
                    reopened.get_logs(type="evaluation", status="success"), {}
                )
                generated = reopened.get_record(experiment.generation_record)
                assert generated and generated.log_location
                self.assertEqual(generated.status, "success")
                original = Path(generated.log_location).read_bytes()
                Pipeline(experiment, reopened).evaluate(show_progress=False)
                saved = Project("run")
                successful = saved.get_record(experiment.evaluation_record)
                assert successful
                self.assertEqual(successful.status, "success")
                self.assertEqual(Path(generated.log_location).read_bytes(), original)
                result = saved.get_log(experiment.evaluation_record)
                assert result and result.samples and result.samples[0].scores
                rating = next(iter(result.samples[0].scores.values()))
                assert rating.metadata
                self.assertAlmostEqual(cast(float, rating.value), 8.7 / 9)
                self.assertEqual(rating.metadata["estimator"], "weighted_rating_tokens")
                self.assertEqual(judge.generate.await_count, 2)

    def test_legacy_weighted_success_remains_readable_but_cannot_skip_fresh_scoring(
        self,
    ):
        for suffix in ("json", "eval"):
            with (
                self.subTest(suffix=suffix),
                TemporaryDirectory() as tmp,
                patch(
                    "evalsense.workflow.project.PROJECTS_PATH", Path(tmp) / "projects"
                ),
            ):
                experiment, project, judge = weighted_fixture(tmp, [complete_rating()])
                Pipeline(experiment, project).generate(
                    show_progress=False,
                    eval_kwargs={"display": "none", "log_format": suffix},
                )
                evaluator = experiment.evaluator
                assert evaluator and evaluator.cache_parameters
                legacy = replace(
                    evaluator,
                    cache_parameters={
                        key: value
                        for key, value in evaluator.cache_parameters.items()
                        if key not in {"weighting_policy", "estimator_format"}
                    },
                )
                legacy_record = replace(experiment, evaluator=legacy).evaluation_record
                self.assertNotEqual(legacy_record, experiment.evaluation_record)
                generated = project.get_log(experiment.generation_record)
                assert generated and generated.samples
                old_log = generated.model_copy(deep=True)
                assert old_log.samples
                legacy_score = (3.4 - 1) / 9
                old_log.samples[0].scores = {
                    "legacy-rating": Score(
                        value=legacy_score, metadata={"raw_score": 3.4}
                    )
                }
                old_log.results = EvalResults(
                    total_samples=1,
                    completed_samples=1,
                    scores=[
                        EvalScore(
                            name="legacy-rating",
                            scorer="legacy-rating",
                            metrics={
                                "mean": EvalMetric(name="mean", value=legacy_score)
                            },
                        )
                    ],
                )
                project.evaluation_log_path.mkdir(parents=True, exist_ok=True)
                path = project.evaluation_log_path / f"legacy.{suffix}"
                write_eval_log(old_log, location=str(path))
                project.update_record(
                    legacy_record,
                    ResultRecord(status="success", log_location=str(path)),
                )
                original = path.read_bytes()
                Pipeline(experiment, Project("run")).evaluate(show_progress=False)
                reopened = Project("run")
                current = reopened.get_record(experiment.evaluation_record)
                previous = reopened.get_record(legacy_record)
                assert current and previous
                self.assertEqual(current.status, "success")
                self.assertEqual(previous.status, "success")
                self.assertEqual(path.read_bytes(), original)
                historical = reopened.get_log(legacy_record)
                assert (
                    historical and historical.samples and historical.samples[0].scores
                )
                self.assertEqual(
                    historical.samples[0].scores["legacy-rating"].value, legacy_score
                )
                fresh = reopened.get_log(experiment.evaluation_record)
                assert fresh and fresh.samples and fresh.samples[0].scores
                rating = next(iter(fresh.samples[0].scores.values()))
                assert rating.metadata
                self.assertAlmostEqual(rating.metadata["raw_score"], 9.7)
                self.assertEqual(rating.metadata["estimator_format"], ESTIMATOR_FORMAT)
                self.assertEqual(judge.generate.await_count, 1)
