import json
from pathlib import Path
from typing import Literal
from unittest.mock import patch
from zipfile import ZipFile

from inspect_ai.log import read_eval_log

from evalsense.evaluation import RecordStatus

from project_log_fixture import ProjectLogFixture


class ProjectLogRetentionTests(ProjectLogFixture):
    def test_only_terminal_failures_are_removed_in_both_formats_and_directories(self):
        statuses: tuple[RecordStatus, ...] = (
            "started", "success", "error", "cancelled"
        )
        retained = {}
        removed = []
        for root in (self.project.generation_log_path, self.project.evaluation_log_path):
            for suffix in (".json", ".eval"):
                for status in statuses:
                    path = self.write_log(
                        root / "attempt" / (status + suffix), status
                    )
                    if status in {"started", "success"}:
                        retained[path] = path.read_bytes()
                    else:
                        removed.append(path)

        self.project.cleanup_incomplete_logs()

        for path, contents in retained.items():
            self.assertEqual(path.read_bytes(), contents)
        for path in removed:
            self.assertFalse(path.exists())

    def test_malformed_files_do_not_prevent_cleanup_of_readable_siblings(self):
        root = self.project.generation_log_path
        root.mkdir()
        bad_json = root / "a-corrupt.json"
        bad_eval = root / "b-truncated.eval"
        bad_json.write_bytes(b'{"status": "error",')
        bad_eval.write_bytes(b"PK")
        retained = {path: path.read_bytes() for path in (bad_json, bad_eval)}
        failed = self.write_log(root / "z-failure.json", "error")

        self.project.cleanup_incomplete_logs()

        self.assertFalse(failed.exists())
        for path, contents in retained.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_missing_and_unknown_statuses_are_preserved(self):
        paths = []
        for status in (None, "unknown"):
            path = self.write_log(
                self.project.generation_log_path / f"status-{status}.json", "error"
            )
            data = json.loads(path.read_text())
            if status is None:
                data.pop("status")
            else:
                data["status"] = status
            path.write_text(json.dumps(data))
            paths.append(path)
        contents = {path: path.read_bytes() for path in paths}
        self.assertEqual(read_eval_log(str(paths[0])).status, "started")

        self.project.cleanup_incomplete_logs()

        self.assertEqual({path: path.read_bytes() for path in paths}, contents)

    def test_corrupt_samples_are_preserved_even_when_the_header_is_readable(self):
        path = self.write_log(self.project.generation_log_path / "corrupt.eval", "error")
        with ZipFile(path, "a") as archive:
            archive.writestr("samples/1.json", "{")
        self.assertEqual(read_eval_log(str(path), header_only=True).status, "error")
        with self.assertRaises(json.JSONDecodeError):
            read_eval_log(str(path))
        contents = path.read_bytes()

        self.project.cleanup_incomplete_logs()

        self.assertEqual(path.read_bytes(), contents)

    def test_unreadable_file_is_retained_while_a_sibling_is_cleaned(self):
        unreadable = self.write_log(
            self.project.generation_log_path / "a-unreadable.json", "error"
        )
        sibling = self.write_log(
            self.project.generation_log_path / "z-failure.json", "error"
        )
        contents = unreadable.read_bytes()

        def read_or_deny(location):
            if Path(location) == unreadable:
                raise PermissionError("file is temporarily unreadable")
            return read_eval_log(location)

        with patch("evalsense.workflow.project.read_eval_log", side_effect=read_or_deny):
            self.project.cleanup_incomplete_logs()

        self.assertEqual(unreadable.read_bytes(), contents)
        self.assertFalse(sibling.exists())

    def test_failure_to_remove_one_file_does_not_prevent_sibling_cleanup(self):
        retained = self.write_log(
            self.project.generation_log_path / "a-retained.json", "error"
        )
        sibling = self.write_log(
            self.project.generation_log_path / "z-failure.json", "error"
        )
        contents = retained.read_bytes()
        unlink = Path.unlink

        def unlink_or_deny(path, *args, **kwargs):
            if path == retained:
                raise PermissionError("file cannot be removed")
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", new=unlink_or_deny):
            self.project.cleanup_incomplete_logs()

        self.assertEqual(retained.read_bytes(), contents)
        self.assertFalse(sibling.exists())

    def test_compatibility_getter_reads_nested_logs_without_classifying_status(self):
        statuses: tuple[RecordStatus, ...] = ("error", "started", "success")
        paths = [
            self.write_log(
                self.project.generation_log_path / "attempt" / (status + ".json"),
                status,
            )
            for status in statuses
        ]

        logs = self.project.get_incomplete_logs("generation")

        self.assertEqual([Path(log.location) for log in logs], sorted(paths))
        self.assertEqual({log.status for log in logs}, set(statuses))
        self.assertTrue(all(path.exists() for path in paths))

    def test_path_inventory_remains_available_when_compatibility_getter_cannot_read(self):
        path = self.project.evaluation_log_path / "partial.json"
        path.parent.mkdir()
        path.write_text("{")

        with self.assertRaises(ValueError):
            self.project.get_incomplete_logs("evaluation")

        self.assertEqual(self.project.get_untracked_log_paths("evaluation"), [path])
        self.assertEqual(path.read_text(), "{")

    def test_explicit_cleanup_does_not_create_missing_log_directories(self):
        types: tuple[Literal["generation", "evaluation"], ...] = (
            "generation", "evaluation"
        )
        self.project.cleanup_incomplete_logs()

        for kind in types:
            self.assertEqual(self.project.get_untracked_log_paths(kind), [])
        self.assertFalse(self.project.generation_log_path.exists())
        self.assertFalse(self.project.evaluation_log_path.exists())
