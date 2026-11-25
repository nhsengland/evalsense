import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, write_eval_log
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import Evaluator, ExperimentConfig, GenerationSource, ResultRecord
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.workflow import Pipeline, Project


class EvaluationSourceTests(unittest.TestCase):
    def setUp(self):
        storage = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch("evalsense.workflow.project.PROJECTS_PATH", storage))
        self.project = Project("source")
        self.experiment = ExperimentConfig(
            HuggingFaceDatasetManager("fixture", version="1", splits=["test"]),
            GenerationSteps("generate", generate()),
            ModelConfig("mockllm/model"),
            evaluator=Evaluator("exact", exact()),
        )
        self.generation_key = self.experiment.generation_record
        self.evaluation_key = self.experiment.evaluation_record

    def save_generation(self, name="2026_task_first.json", marker="first"):
        path = self.project.generation_log_path / name
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
                    metadata={"marker": marker},
                ),
                samples=[],
            ),
            location=str(path),
        )
        self.project.update_record(
            self.generation_key,
            ResultRecord(status="success", log_location=str(path)),
        )
        return path

    def initialise(self):
        result = self.project.get_record(
            self.evaluation_key, init_eval_record_from_generations=True
        )
        assert result is not None and result.log_location is not None
        return result

    def test_source_is_saved_and_survives_scoring_and_reopening(self):
        for suffix in ("json", "eval"):
            with self.subTest(suffix=suffix):
                self.project = Project("source-" + suffix)
                path = self.save_generation("2026_task_first." + suffix)
                contents = path.read_bytes()
                initial = self.initialise()
                expected = GenerationSource(
                    log_location=str(path.resolve()),
                    sha256=hashlib.sha256(contents).hexdigest(),
                )
                self.assertEqual(initial.source_generation, expected)
                with patch(
                    "evalsense.workflow.pipeline.score", side_effect=lambda log, **kw: log
                ):
                    Pipeline(self.experiment, self.project).evaluate(show_progress=False)
                saved = Project(self.project.name).get_record(self.evaluation_key)
                assert saved is not None
                self.assertEqual(saved.status, "success")
                self.assertEqual(saved.source_generation, expected)
                self.assertEqual(path.read_bytes(), contents)

    def test_old_metadata_without_source_remains_readable(self):
        self.save_generation()
        initial = self.initialise()
        metadata_path = self.project.project_path / Project.METADATA_FILE
        metadata = json.loads(metadata_path.read_text())
        for _, record in metadata["evaluation"]:
            record.pop("source_generation")
        metadata_path.write_text(json.dumps(metadata))
        reopened = Project(self.project.name).get_record(self.evaluation_key)
        assert reopened is not None
        self.assertIsNone(reopened.source_generation)
        self.assertEqual(reopened.log_location, initial.log_location)

    def test_manual_evaluator_names_cannot_overwrite_another_results_log(self):
        self.save_generation()
        first_key = self.evaluation_key.model_copy(
            update={"evaluator_name": "a b", "evaluator_fingerprint": ""}
        )
        second_key = first_key.model_copy(update={"evaluator_name": "a-b"})
        first = self.project.get_record(first_key, init_eval_record_from_generations=True)
        assert first is not None and first.log_location
        first_path = Path(first.log_location)
        contents = first_path.read_bytes() + b"\n"
        first_path.write_bytes(contents)
        completed = first.model_copy(update={"status": "success"})
        self.project.update_record(first_key, completed)
        second = self.project.get_record(second_key, init_eval_record_from_generations=True)
        assert second is not None
        self.assertNotEqual(second.log_location, first.log_location)
        self.assertEqual(self.project.get_record(first_key), completed)
        self.assertEqual(first_path.read_bytes(), contents)

    def test_long_generation_filename_uses_a_bounded_evaluation_name(self):
        path = self.save_generation("2026_task_" + "x" * 150 + ".json")
        result = self.initialise()
        assert result.log_location
        self.assertLessEqual(len(Path(result.log_location).name.encode("utf-8")), 255)
        self.assertEqual(result.source_generation, GenerationSource.capture(path))

    def test_mismatched_copy_cannot_receive_source_provenance(self):
        self.save_generation()

        def wrong_copy(source, destination):
            Path(destination).write_text("unrelated bytes")

        with patch("evalsense.workflow.project.shutil.copy", side_effect=wrong_copy):
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.initialise()
        self.assertNotIn(self.evaluation_key, self.project.records.evaluation)
        self.assertEqual(Project(self.project.name).records.evaluation, {})


    def test_new_generation_refreshes_an_existing_success(self):
        first_path = self.save_generation()
        first = self.initialise()
        self.project.update_record(
            self.evaluation_key, first.model_copy(update={"status": "success"})
        )
        second_path = self.save_generation("2026_task_second.json", "second")
        refreshed = self.initialise()
        self.assertEqual(refreshed.status, "started")
        self.assertEqual(refreshed.source_generation, GenerationSource.capture(second_path))
        self.assertNotEqual(refreshed.source_generation, first.source_generation)
        log = self.project.get_log(self.evaluation_key)
        assert log is not None
        self.assertEqual(log.eval.metadata, {"marker": "second"})
        self.assertNotEqual(str(first_path), str(second_path))

    def test_replaced_source_at_the_same_path_is_refreshed(self):
        path = self.save_generation()
        first = self.initialise()
        self.save_generation(marker="replacement")
        refreshed = self.initialise()
        self.assertEqual(refreshed.source_generation, GenerationSource.capture(path))
        self.assertNotEqual(refreshed.source_generation, first.source_generation)
        log = self.project.get_log(self.evaluation_key)
        assert log is not None
        self.assertEqual(log.eval.metadata, {"marker": "replacement"})

    def test_unchanged_source_reuses_success_without_copying(self):
        self.save_generation()
        first = self.initialise().model_copy(update={"status": "success"})
        self.project.update_record(self.evaluation_key, first)
        with patch("evalsense.workflow.project.shutil.copy") as copy:
            self.assertEqual(self.initialise(), first)
        copy.assert_not_called()

    def test_legacy_success_is_refreshed_once(self):
        path = self.save_generation()
        legacy = self.initialise().model_copy(
            update={"status": "success", "source_generation": None}
        )
        self.project.update_record(self.evaluation_key, legacy)
        self.project = Project(self.project.name)
        self.assertEqual(self.project.get_record(self.evaluation_key), legacy)
        refreshed = self.initialise()
        self.assertEqual(refreshed.status, "started")
        self.assertEqual(refreshed.source_generation, GenerationSource.capture(path))
        completed = refreshed.model_copy(update={"status": "success"})
        self.project.update_record(self.evaluation_key, completed)
        self.project = Project(self.project.name)
        self.assertEqual(self.initialise(), completed)

    def test_each_evaluator_refreshes_its_own_copy(self):
        self.save_generation()
        self.initialise()
        other_key = self.evaluation_key.model_copy(update={"evaluator_name": "other"})
        other = self.project.get_record(other_key, init_eval_record_from_generations=True)
        assert other is not None
        path = self.save_generation("2026_task_second.json", "second")
        first_new = self.initialise()
        self.assertEqual(self.project.get_record(other_key), other)
        other_new = self.project.get_record(other_key, init_eval_record_from_generations=True)
        assert other_new is not None
        self.assertEqual(first_new.source_generation, GenerationSource.capture(path))
        self.assertEqual(other_new.source_generation, first_new.source_generation)
        self.assertNotEqual(other_new.log_location, first_new.log_location)


if __name__ == "__main__":
    unittest.main()
