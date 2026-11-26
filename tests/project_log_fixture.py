from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, write_eval_log
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import Evaluator, ExperimentConfig, RecordStatus
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.workflow import Project


class ProjectLogFixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch("evalsense.workflow.project.PROJECTS_PATH", self.root))
        self.project = Project("logs")
        self.experiment = ExperimentConfig(
            HuggingFaceDatasetManager("fixture", version="1", splits=["test"]),
            GenerationSteps("generate", generate()),
            ModelConfig("mockllm/model"),
            evaluator=Evaluator("exact", exact()),
        )

    def write_log(
        self,
        path: Path,
        status: RecordStatus = "success",
        *,
        task: str = "generate",
        metadata: dict | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(
            EvalLog(
                status=status,
                eval=EvalSpec(
                    created="2026-01-01T00:00:00Z",
                    task=task,
                    model="mockllm/model",
                    config=EvalConfig(),
                    dataset=EvalDataset(),
                    metadata=metadata,
                ),
                samples=[],
            ),
            location=str(path),
        )
        return path
