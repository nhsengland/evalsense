from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, write_eval_log
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import Evaluator, ExperimentConfig, ResultRecord
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.workflow import Pipeline, Project


class EvaluationGenerationStatusTests(unittest.TestCase):
    def setUp(self):
        storage = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch("evalsense.workflow.project.PROJECTS_PATH", storage))
        self.project = Project("test")
        self.experiment = ExperimentConfig(
            HuggingFaceDatasetManager("fixture", version="1", splits=["test"]),
            GenerationSteps("generate", generate()),
            ModelConfig("mockllm/model"),
            evaluator=Evaluator("exact", exact()),
        )
        self.generate_key = self.experiment.generation_record
        self.evaluate_key = self.experiment.evaluation_record
        self.score = self.enterContext(
            patch(
                "evalsense.workflow.pipeline.score", side_effect=lambda log, **kw: log
            )
        )

    def write_log(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(
            EvalLog(
                status="success",
                eval=EvalSpec(
                    created="2026-01-01T00:00:00Z",
                    task="task",
                    model="model",
                    dataset=EvalDataset(),
                    config=EvalConfig(),
                ),
                samples=[],
            ),
            location=str(path),
        )

    def evaluate(self, project, force=False):
        Pipeline(self.experiment, project).evaluate(
            show_progress=False, force_rerun=force
        )

    def test_only_successful_generations_are_eligible_for_scoring(self):
        path = self.project.generation_log_path / "2026_task_run.json"
        self.write_log(path)
        for status in ("error", "started", "cancelled"):
            with self.subTest(status=status):
                self.project.update_record(
                    self.generate_key,
                    ResultRecord(status=status, log_location=str(path)),
                )
                self.evaluate(self.project, force=True)
                self.score.assert_not_called()
                self.assertNotIn(self.evaluate_key, self.project.records.evaluation)

    def test_stored_alias_rebuilds_from_new_successful_generation(self):
        old = self.project.generation_log_path / "2026_task_previous.json"
        self.write_log(old)
        failure = ResultRecord(status="error", log_location=str(old))
        self.project.update_record(self.generate_key, failure)
        self.project.update_record(self.evaluate_key, failure)

        project = Project("test")
        self.assertEqual(project.get_record(self.evaluate_key), failure)
        self.evaluate(project, force=True)
        self.score.assert_not_called()

        new = project.generation_log_path / "attempt" / "2026_task_next.json"
        self.write_log(new)
        project.update_record(
            self.generate_key, ResultRecord(status="success", log_location=str(new))
        )
        self.evaluate(project)

        self.score.assert_called_once()
        scored_path = Path(self.score.call_args.kwargs["log"].location)
        self.assertEqual(scored_path.parent, project.evaluation_log_path)
        self.assertIn("next", scored_path.name)
        self.assertTrue(new.exists())
        self.assertFalse(old.exists())

    def test_evaluation_owned_error_can_retry_after_reload(self):
        path = self.project.generation_log_path / "2026_task_input.json"
        self.write_log(path)
        self.project.update_record(
            self.generate_key, ResultRecord(status="success", log_location=str(path))
        )
        evaluation = self.project.get_record(
            self.evaluate_key, init_eval_record_from_generations=True
        )
        assert evaluation is not None
        self.project.update_record(
            self.evaluate_key, evaluation.model_copy(update={"status": "error"})
        )

        self.evaluate(Project("test"))

        self.score.assert_called_once()
        self.assertEqual(
            self.score.call_args.kwargs["log"].location, evaluation.log_location
        )
