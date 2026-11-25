"""Invalid judge responses must remain failures in saved project records."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
import unittest
from unittest.mock import AsyncMock, Mock, patch

from datasets import Dataset, DatasetDict
from inspect_ai.log import EvalMetric, EvalResults, EvalScore, write_eval_log
from inspect_ai.model import Model, ModelOutput
from inspect_ai.scorer import Score
from inspect_ai.solver import generate

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import ExperimentConfig, ResultRecord
from evalsense.evaluation.evaluators.g_eval import (
    GEvalScorerFactory,
    get_g_eval_evaluator,
)
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.workflow import Pipeline, Project


def fixture(tmp: str, responses: list[str]):
    manager = HuggingFaceDatasetManager(
        "fixture", version="1", splits=["test"], data_dir=tmp
    )
    DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "id": [1],
                    "input": ["synthetic note"],
                    "target": ["reference"],
                }
            )
        }
    ).save_to_disk(manager.main_data_path)
    judge = Mock(spec=Model)
    judge.generate = AsyncMock(
        side_effect=[ModelOutput.from_content("fixture", text) for text in responses]
    )
    evaluator = get_g_eval_evaluator(
        prompt_template="Rate {prediction}",
        model_config=ModelConfig("mockllm/judge"),
        logprobs=False,
    )
    evaluator.scorer = cast(GEvalScorerFactory, evaluator.scorer).create_scorer(
        cast(Model, judge)
    )
    evaluator.model_config = None
    experiment = ExperimentConfig(
        manager,
        GenerationSteps("generate", generate()),
        ModelConfig("mockllm/model"),
        evaluator=evaluator,
    )
    return experiment, Project("run"), judge


class GEvalPersistenceTests(unittest.TestCase):
    def test_error_survives_reopening_and_a_valid_retry_preserves_generated_outputs(
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
                experiment, project, judge = fixture(tmp, ["0/10", "8"])
                Pipeline(experiment, project).run(
                    show_progress=False,
                    eval_kwargs={"display": "none", "log_format": suffix},
                )
                reopened = Project("run")
                failed = reopened.get_record(experiment.evaluation_record)
                assert failed
                self.assertEqual(failed.status, "error")
                self.assertIn("complete integer", failed.error_message or "")
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
                self.assertAlmostEqual(cast(float, rating.value), 7 / 9)
                self.assertEqual(judge.generate.await_count, 2)

    def test_legacy_success_is_preserved_but_cannot_satisfy_the_strict_evaluator(self):
        with (
            TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp) / "projects"),
        ):
            experiment, project, judge = fixture(tmp, ["0/10"])
            Pipeline(experiment, project).generate(
                show_progress=False,
                eval_kwargs={"display": "none", "log_format": "json"},
            )
            evaluator = experiment.evaluator
            assert evaluator and evaluator.cache_parameters
            legacy = replace(
                evaluator,
                cache_parameters={
                    key: value
                    for key, value in evaluator.cache_parameters.items()
                    if key != "score_format"
                },
            )
            legacy_record = replace(experiment, evaluator=legacy).evaluation_record
            generated = project.get_log(experiment.generation_record)
            assert generated and generated.samples
            old_log = generated.model_copy(deep=True)
            assert old_log.samples
            old_log.samples[0].scores = {"legacy-rating": Score(value=1.0)}
            old_log.results = EvalResults(
                total_samples=1,
                completed_samples=1,
                scores=[
                    EvalScore(
                        name="legacy-rating",
                        scorer="legacy-rating",
                        metrics={
                            "mean": EvalMetric(name="mean", value=1.0),
                        },
                    )
                ],
            )
            project.evaluation_log_path.mkdir(parents=True, exist_ok=True)
            path = project.evaluation_log_path / "legacy.json"
            write_eval_log(old_log, location=str(path))
            project.update_record(
                legacy_record, ResultRecord(status="success", log_location=str(path))
            )
            original = path.read_bytes()
            Pipeline(experiment, project).evaluate(show_progress=False)
            reopened = Project("run")
            current = reopened.get_record(experiment.evaluation_record)
            previous = reopened.get_record(legacy_record)
            assert current and previous
            self.assertEqual(current.status, "error")
            self.assertEqual(previous.status, "success")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(judge.generate.await_count, 1)
