from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import Evaluator, ExperimentConfig, ResultRecord
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.workflow import Project


class SharedProjectLogTests(unittest.TestCase):
    def setUp(self):
        storage = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch("evalsense.workflow.project.PROJECTS_PATH", storage))
        self.project = Project("legacy")
        self.experiment = ExperimentConfig(
            HuggingFaceDatasetManager("fixture", version="1", splits=["test"]),
            GenerationSteps("generate", generate()),
            ModelConfig("mockllm/model"),
            evaluator=Evaluator("exact", exact()),
        )
        self.project.generation_log_path.mkdir()
        self.path = self.project.generation_log_path / "2026_task_run.json"
        self.path.write_bytes(b"shared log")
        result = ResultRecord(status="error", log_location=str(self.path))
        self.project.update_record(self.experiment.generation_record, result)
        self.project.update_record(self.experiment.evaluation_record, result)

    def test_removing_evaluation_preserves_generation_log(self):
        self.project.remove_record(self.experiment.evaluation_record)
        self.assertEqual(self.path.read_bytes(), b"shared log")

        self.project.remove_record(self.experiment.generation_record)
        self.assertFalse(self.path.exists())

    def test_replacing_generation_preserves_evaluation_log_on_reload(self):
        self.project.update_record(self.experiment.generation_record, ResultRecord())

        reopened = Project("legacy")
        evaluation = reopened.get_record(self.experiment.evaluation_record)
        self.assertIsNotNone(evaluation)
        self.assertEqual(self.path.read_bytes(), b"shared log")

        reopened.remove_record(self.experiment.evaluation_record)
        self.assertFalse(self.path.exists())
