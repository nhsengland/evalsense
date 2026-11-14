from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, write_eval_log

from evalsense.workflow import Project


class GenerationLogRecoveryTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch("evalsense.workflow.project.PROJECTS_PATH", root))
        self.project = Project("experiment")
        self.generation_id = "a" * 64
        self.generation_dir = self.project.generation_log_path / self.generation_id
        self.attempt = self.generation_dir / ("b" * 32)
        self.attempt.mkdir(parents=True)
        self.metadata = {"evalsense_generation_id": self.generation_id}

    def write_log(self, path, task="generate", metadata=None):
        log = EvalLog(
            status="success",
            eval=EvalSpec(
                created="2026-01-01T00:00:00Z",
                task=task,
                model="mockllm/model",
                config=EvalConfig(),
                dataset=EvalDataset(),
                metadata=metadata,
            ),
            samples=[],
        )
        write_eval_log(log, location=str(path))

    def recover(self):
        return self.project._recover_generation_log(
            self.attempt, "generate", self.generation_id
        )

    def test_one_owned_log_in_either_format(self):
        for suffix in (".eval", ".json"):
            with self.subTest(suffix=suffix):
                path = self.attempt / f"log{suffix}"
                self.write_log(path, metadata=self.metadata)
                log = self.recover()
                self.assertIsNotNone(log)
                assert log is not None
                self.assertEqual(Path(log.location), path)
                path.unlink()

    def test_requires_task_and_generation_identity(self):
        path = self.attempt / "log.json"
        for task, metadata in (
            ("other", self.metadata),
            ("generate", None),
            ("generate", {"evalsense_generation_id": "c" * 64}),
        ):
            with self.subTest(task=task, metadata=metadata):
                self.write_log(path, task=task, metadata=metadata)
                self.assertIsNone(self.recover())
                path.unlink()

    def test_ignores_other_attempts_and_rejects_unreadable_or_ambiguous_logs(self):
        other = self.generation_dir / ("d" * 32)
        other.mkdir()
        self.write_log(other / "log.json", metadata=self.metadata)
        self.assertIsNone(self.recover())

        (self.attempt / "unreadable.json").write_text("incomplete")
        self.assertIsNone(self.recover())
        self.write_log(self.attempt / "log.eval", metadata=self.metadata)
        self.assertIsNone(self.recover())

    def test_ignores_link_to_another_attempt(self):
        other = self.generation_dir / ("d" * 32)
        other.mkdir()
        source = other / "log.json"
        self.write_log(source, metadata=self.metadata)
        (self.attempt / "log.json").symlink_to(source)
        self.assertIsNone(self.recover())
