from typing import Literal, cast
from unittest.mock import patch

from evalsense.evaluation import GenerationSource, ResultRecord

from project_log_fixture import ProjectLogFixture


class ProjectLogInventoryTests(ProjectLogFixture):
    def test_missing_log_directories_are_not_created(self):
        self.assertEqual(self.project.get_untracked_log_paths("generation"), [])
        self.assertEqual(self.project.get_untracked_log_paths("evaluation"), [])
        self.assertFalse(self.project.generation_log_path.exists())
        self.assertFalse(self.project.evaluation_log_path.exists())

    def test_lists_supported_files_in_root_and_nested_attempts_in_path_order(self):
        root = self.project.generation_log_path
        paths = [
            root / "z.json",
            root / ("a" * 64) / ("b" * 32) / "log.eval",
            root / "a.json",
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"partial")
        (root / "unfinished.tmp").write_bytes(b"partial")
        (root / "directory.json").mkdir()
        other = self.write_log(self.project.evaluation_log_path / "evaluation.eval")

        self.assertEqual(
            self.project.get_untracked_log_paths("generation"), sorted(paths)
        )
        self.assertEqual(
            self.project.get_untracked_log_paths("evaluation"), [other]
        )

    def test_inspection_does_not_parse_logs_or_change_files_and_metadata(self):
        path = self.project.generation_log_path / "corrupt.eval"
        path.parent.mkdir()
        path.write_bytes(b"not a log")
        metadata = self.project.project_path / self.project.METADATA_FILE
        before = {item: item.read_bytes() for item in (path, metadata)}

        with patch(
            "evalsense.workflow.project.read_eval_log",
            side_effect=AssertionError("Inspection must not parse logs"),
        ):
            self.assertEqual(
                self.project.get_untracked_log_paths("generation"), [path]
            )

        self.assertEqual({item: item.read_bytes() for item in before}, before)

    def test_registered_logs_are_excluded(self):
        generation = self.write_log(self.project.generation_log_path / "saved.json")
        evaluation = self.write_log(self.project.evaluation_log_path / "saved.eval")
        self.project.update_record(
            self.experiment.generation_record,
            ResultRecord(status="success", log_location=str(generation)),
        )
        self.project.update_record(
            self.experiment.evaluation_record,
            ResultRecord(status="success", log_location=str(evaluation)),
        )

        self.assertEqual(self.project.get_untracked_log_paths("generation"), [])
        self.assertEqual(self.project.get_untracked_log_paths("evaluation"), [])

    def test_invalid_log_type_is_rejected(self):
        invalid = cast(Literal["generation", "evaluation"], "other")
        with self.assertRaisesRegex(ValueError, "Invalid log type: other"):
            self.project.get_untracked_log_paths(invalid)

    def test_generation_sources_are_excluded_without_a_direct_record(self):
        path = self.write_log(self.project.generation_log_path / "source.json", "error")
        contents = path.read_bytes()
        self.project.update_record(
            self.experiment.evaluation_record,
            ResultRecord(
                status="started", source_generation=GenerationSource.capture(path)
            ),
        )

        self.assertEqual(self.project.get_untracked_log_paths("generation"), [])
        self.assertEqual(path.read_bytes(), contents)
