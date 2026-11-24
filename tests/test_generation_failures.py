from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, write_eval_log
from inspect_ai.solver import generate

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import ExperimentConfig, ResultRecord
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.workflow import Pipeline, Project


class GenerationFailureTests(unittest.TestCase):
    def setUp(self):
        storage = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch("evalsense.workflow.project.PROJECTS_PATH", storage))
        self.project = Project("fixture")
        self.project.generation_log_path.mkdir()
        self.experiment = ExperimentConfig(
            HuggingFaceDatasetManager("fixture", version="1", splits=["test"]),
            GenerationSteps("generate", generate()),
            ModelConfig("mockllm/model"),
        )
        self.key = self.experiment.generation_record
        self.pipeline = Pipeline(self.experiment, self.project)
        self.dataset = MemoryDataset([Sample(input="sample")])

    def generate(self):
        self.pipeline._generate_on_dataset(self.experiment, self.dataset, False, {}, {})

    def write_log(self, name):
        path = self.project.generation_log_path / name
        log = EvalLog(
            status="success",
            eval=EvalSpec(
                created="2026-01-01T00:00:00Z", task="other", model="mockllm/model",
                dataset=EvalDataset(), config=EvalConfig(),
            ),
        )
        write_eval_log(log, location=str(path))
        return path

    def test_fresh_failure_preserves_error_with_unrelated_success_present(self):
        other = self.write_log("other.json")
        with patch("evalsense.workflow.pipeline.eval", side_effect=ValueError("failed")):
            self.generate()

        self.assertEqual(
            self.project.get_record(self.key),
            ResultRecord(status="error", error_message="failed"),
        )
        self.assertTrue(other.exists())

    def test_interrupted_retry_records_error_and_propagates_interrupt(self):
        previous = self.write_log("previous.json")
        self.project.update_record(
            self.key, ResultRecord(status="error", log_location=str(previous))
        )
        interruption = KeyboardInterrupt("stop retry")
        with (
            patch("evalsense.workflow.pipeline.eval_retry", side_effect=interruption),
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            self.generate()

        self.assertIs(raised.exception, interruption)
        self.assertEqual(
            self.project.get_record(self.key),
            ResultRecord(status="error", error_message="stop retry"),
        )
        self.assertTrue(previous.exists())
