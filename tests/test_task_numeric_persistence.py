from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evalsense.evaluation import ExperimentConfig
from evalsense.tasks import TaskPreprocessor
from evalsense.utils.cache import fingerprint
from evalsense.utils.files import to_safe_filename
from evalsense.workflow import Project
from test_task_numeric_pipeline import (
    numeric_dataset,
    numeric_experiment,
    reported_accuracy,
    run_numeric_experiment,
)


def legacy_preprocessor_key(preprocessor: TaskPreprocessor) -> str:
    # Exact identity used before the precision-preserving transport was added.
    return fingerprint(
        {
            "implementation": type(preprocessor),
            "function": preprocessor.preprocessing_function,
        },
        version=preprocessor.version,
    )


class NumericPersistenceTests(unittest.TestCase):
    def assert_preserved_logs(
        self, project: Project, experiment: ExperimentConfig, values: list[float]
    ):
        paths: dict[Path, bytes] = {}
        for key in (experiment.generation_record, experiment.evaluation_record):
            result = project.get_record(key)
            assert result is not None and result.log_location is not None
            self.assertEqual(result.status, "success")
            path = Path(result.log_location)
            paths[path] = path.read_bytes()
            log = project.get_log(key)
            assert log is not None and log.samples is not None
            samples = sorted(log.samples, key=lambda sample: int(str(sample.id)))
            self.assertEqual(len(samples), len(values))
            for sample, value in zip(samples, values, strict=True):
                assert isinstance(sample.target, str) and sample.metadata is not None
                self.assertEqual(float(sample.target).hex(), value.hex())
                self.assertEqual(sample.metadata["measured"].hex(), value.hex())
                self.assertEqual(float(sample.output.completion).hex(), value.hex())
        log = project.get_log(experiment.evaluation_record)
        assert log is not None
        self.assertEqual(reported_accuracy(log), 1.0)
        return paths

    def test_saved_numbers_and_scores_survive_reopening_and_cached_runs(self):
        for log_format in ("json", "eval"):
            with (
                self.subTest(log_format=log_format),
                tempfile.TemporaryDirectory() as tmp,
                patch(
                    "evalsense.workflow.project.PROJECTS_PATH", Path(tmp) / "projects"
                ),
            ):
                values = [1e-11, -3e-11, 1.2345678901234567]
                manager = numeric_dataset(tmp, values)
                calls: list[str] = []
                experiment = numeric_experiment(manager, "faithful", calls)
                project = Project("numeric-persistence")
                run_numeric_experiment(experiment, project, log_format)
                paths = self.assert_preserved_logs(project, experiment, values)
                self.assertEqual({p.suffix for p in paths}, {"." + log_format})

                reopened = Project(project.name)
                self.assertEqual(
                    self.assert_preserved_logs(reopened, experiment, values), paths
                )
                run_numeric_experiment(experiment, reopened, log_format)
                self.assertEqual(len(calls), len(values))
                self.assertEqual(
                    self.assert_preserved_logs(
                        Project(project.name), experiment, values
                    ),
                    paths,
                )

    def test_rounded_legacy_results_remain_readable_without_being_reused(self):
        for log_format in ("json", "eval"):
            with (
                self.subTest(log_format=log_format),
                tempfile.TemporaryDirectory() as tmp,
                patch(
                    "evalsense.workflow.project.PROJECTS_PATH", Path(tmp) / "projects"
                ),
            ):
                values = [1e-11, 2e-11, 4e-11]
                manager = numeric_dataset(tmp, values)
                loaded = manager.load(retrieve=False)
                calls: list[str] = []
                experiment = numeric_experiment(manager, "faithful", calls)
                project = Project("numeric-migration")
                with patch.object(
                    TaskPreprocessor, "cache_key", property(legacy_preprocessor_key)
                ):
                    legacy_generation = experiment.generation_record
                    legacy_evaluation = experiment.evaluation_record
                    data_key = fingerprint(
                        {
                            "dataset": manager.record,
                            "content": getattr(loaded, "_fingerprint"),
                            "preprocessor": experiment.task_preprocessor.cache_key,
                        }
                    )
                    tasks = manager.version_path / "tasks"
                    tasks.mkdir(parents=True, exist_ok=True)
                    name = to_safe_filename(experiment.task_preprocessor.name)[:40]
                    legacy_task = tasks / f"{name}-{data_key}.jsonl"
                    # Recreate the previous pandas-backed writer and its rounded data.
                    loaded.to_json(legacy_task, lines=True)
                    legacy_log = run_numeric_experiment(experiment, project, log_format)
                    self.assertEqual(reported_accuracy(legacy_log), 0.0)
                    assert legacy_log.samples is not None
                    self.assertEqual(
                        [sample.target for sample in legacy_log.samples], ["0.0"] * 3
                    )
                    legacy_paths = {legacy_task: legacy_task.read_bytes()}
                    for key in (legacy_generation, legacy_evaluation):
                        result = project.get_record(key)
                        assert result is not None and result.log_location is not None
                        path = Path(result.log_location)
                        legacy_paths[path] = path.read_bytes()

                self.assertNotEqual(experiment.generation_record, legacy_generation)
                self.assertNotEqual(experiment.evaluation_record, legacy_evaluation)
                self.assertEqual(
                    experiment.generation_record.generator_fingerprint,
                    legacy_generation.generator_fingerprint,
                )
                self.assertEqual(
                    experiment.evaluation_record.evaluator_fingerprint,
                    legacy_evaluation.evaluator_fingerprint,
                )
                reopened = Project(project.name)
                self.assertIsNone(reopened.get_record(experiment.generation_record))
                self.assertIsNone(reopened.get_record(experiment.evaluation_record))
                run_numeric_experiment(experiment, reopened, log_format)
                self.assertEqual(len(calls), 2 * len(values))
                self.assert_preserved_logs(reopened, experiment, values)
                self.assertEqual(len(reopened.records.generation), 2)
                self.assertEqual(len(reopened.records.evaluation), 2)
                self.assertEqual(len(list(tasks.glob("*.jsonl"))), 2)

                final = Project(project.name)
                old_log = final.get_log(legacy_evaluation)
                assert old_log is not None
                self.assertEqual(reported_accuracy(old_log), 0.0)
                self.assert_preserved_logs(final, experiment, values)
                for path, contents in legacy_paths.items():
                    self.assertEqual(path.read_bytes(), contents)


if __name__ == "__main__":
    unittest.main()
