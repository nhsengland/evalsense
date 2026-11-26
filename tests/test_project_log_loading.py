from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalSpec,
    read_eval_log,
    write_eval_log,
)

from evalsense.workflow import Project


class ProjectLogLoadingTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch("evalsense.workflow.project.PROJECTS_PATH", root))

    def test_reopening_preserves_unregistered_successful_generation_logs(self):
        for suffix in (".json", ".eval"):
            with self.subTest(suffix=suffix):
                project = Project("generation-" + suffix[1:])
                path = project.generation_log_path / ("saved" + suffix)
                path.parent.mkdir()
                write_eval_log(
                    EvalLog(
                        status="success",
                        eval=EvalSpec(
                            created="2026-01-01T00:00:00Z",
                            task="generate",
                            model="mockllm/model",
                            config=EvalConfig(),
                            dataset=EvalDataset(),
                        ),
                        samples=[],
                    ),
                    location=str(path),
                )
                contents = path.read_bytes()
                metadata_path = project.project_path / Project.METADATA_FILE
                metadata = metadata_path.read_bytes()

                reopened = Project(project.name)

                self.assertEqual(path.read_bytes(), contents)
                self.assertEqual(read_eval_log(str(path)).status, "success")
                self.assertEqual(metadata_path.read_bytes(), metadata)
                self.assertEqual(reopened.records.generation, {})
                self.assertEqual(reopened.records.evaluation, {})
