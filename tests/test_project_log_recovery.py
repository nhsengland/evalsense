from pathlib import Path
from unittest.mock import patch

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log

from evalsense.utils.cache import fingerprint
from evalsense.utils.files import to_safe_filename
from evalsense.workflow import Pipeline, Project

from project_log_fixture import ProjectLogFixture


class ReopenedGenerationRecoveryTests(ProjectLogFixture):
    def test_log_written_before_metadata_failure_remains_recoverable_after_reopening(self):
        for suffix in (".json", ".eval"):
            with self.subTest(suffix=suffix):
                self.project = Project("interrupted-" + suffix[1:])
                pipeline = Pipeline(self.experiment, self.project)
                written = []
                update_record = self.project.update_record

                def write_generation(*, tasks, log_dir, **kwargs):
                    path = self.write_log(
                        Path(log_dir) / ("generation" + suffix),
                        task=tasks.name,
                        metadata=tasks.metadata,
                    )
                    written.append(path)
                    return [read_eval_log(str(path))]

                def fail_completed_record(key, result, **kwargs):
                    if result.status == "success":
                        raise OSError("metadata update interrupted")
                    return update_record(key, result, **kwargs)

                with (
                    patch(
                        "evalsense.workflow.pipeline.eval",
                        side_effect=write_generation,
                    ),
                    patch.object(
                        self.project, "update_record", side_effect=fail_completed_record
                    ),
                    self.assertRaisesRegex(OSError, "metadata update interrupted"),
                ):
                    pipeline._generate_on_dataset(
                        self.experiment,
                        MemoryDataset([Sample(input="sample")]),
                        False,
                        {},
                        {},
                    )

                self.assertEqual(len(written), 1)
                path = written[0]
                contents = path.read_bytes()
                metadata_path = self.project.project_path / Project.METADATA_FILE
                metadata = metadata_path.read_bytes()

                reopened = Project(self.project.name)
                record = reopened.get_record(self.experiment.generation_record)
                recovered = reopened._recover_generation_log(
                    path.parent,
                    to_safe_filename(self.experiment.generation_record.label),
                    fingerprint(self.experiment.generation_record),
                )

                self.assertEqual(path.read_bytes(), contents)
                self.assertEqual(metadata_path.read_bytes(), metadata)
                assert record is not None
                self.assertEqual(record.status, "started")
                self.assertIsNone(record.log_location)
                self.assertEqual(reopened.get_logs("generation", status="success"), {})
                self.assertEqual(reopened.get_untracked_log_paths("generation"), [path])
                assert recovered is not None
                self.assertEqual(Path(recovered.location), path)
                self.assertEqual(recovered.status, "success")

    def test_reopened_attempts_remain_distinct_and_unregistered(self):
        generation_id = fingerprint(self.experiment.generation_record)
        task_name = to_safe_filename(self.experiment.generation_record.label)
        paths = []
        for attempt, suffix in (("a" * 32, ".json"), ("b" * 32, ".eval")):
            paths.append(
                self.write_log(
                    self.project.generation_log_path
                    / generation_id
                    / attempt
                    / ("generation" + suffix),
                    task=task_name,
                    metadata={"evalsense_generation_id": generation_id},
                )
            )
        contents = {path: path.read_bytes() for path in paths}

        reopened = Project(self.project.name)

        for path in paths:
            recovered = reopened._recover_generation_log(
                path.parent, task_name, generation_id
            )
            assert recovered is not None
            self.assertEqual(Path(recovered.location), path)
        self.assertEqual(reopened.records.generation, {})
        self.assertEqual({path: path.read_bytes() for path in paths}, contents)

    def test_ambiguous_attempt_is_preserved_without_selecting_a_result(self):
        generation_id = fingerprint(self.experiment.generation_record)
        task_name = to_safe_filename(self.experiment.generation_record.label)
        attempt = self.project.generation_log_path / generation_id / ("a" * 32)
        paths = [
            self.write_log(
                attempt / ("generation" + suffix),
                task=task_name,
                metadata={"evalsense_generation_id": generation_id},
            )
            for suffix in (".json", ".eval")
        ]
        contents = {path: path.read_bytes() for path in paths}

        reopened = Project(self.project.name)

        self.assertIsNone(
            reopened._recover_generation_log(attempt, task_name, generation_id)
        )
        self.assertEqual(reopened.records.generation, {})
        self.assertEqual(reopened.get_untracked_log_paths("generation"), sorted(paths))
        self.assertEqual({path: path.read_bytes() for path in paths}, contents)
