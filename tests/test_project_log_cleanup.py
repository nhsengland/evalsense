from unittest.mock import patch

from inspect_ai.log import read_eval_log

from project_log_fixture import ProjectLogFixture


class ProjectLogCleanupTests(ProjectLogFixture):
    def test_explicit_cleanup_removes_nested_failure_and_preserves_success(self):
        failed = self.write_log(
            self.project.generation_log_path / "generation" / "attempt" / "failed.eval",
            "error",
        )
        successful = self.write_log(
            self.project.evaluation_log_path / "successful.json"
        )
        contents = successful.read_bytes()

        self.project.cleanup_incomplete_logs()

        self.assertFalse(failed.exists())
        self.assertEqual(successful.read_bytes(), contents)

    def test_symlink_files_and_directories_are_not_followed(self):
        external = self.write_log(self.root / "other-project" / "failure.json", "error")
        contents = external.read_bytes()
        root = self.project.generation_log_path
        root.mkdir()
        link = root / "linked.json"
        link.symlink_to(external)
        directory = root / "linked-attempt"
        directory.symlink_to(external.parent, target_is_directory=True)
        local = self.write_log(root / "local.json", "error")

        self.assertEqual(self.project.get_untracked_log_paths("generation"), [local])
        self.project.cleanup_incomplete_logs()

        self.assertFalse(local.exists())
        self.assertTrue(link.is_symlink())
        self.assertTrue(directory.is_symlink())
        self.assertEqual(external.read_bytes(), contents)

    def test_symlink_log_roots_are_not_followed(self):
        external = self.write_log(self.root / "other-project" / "failure.eval", "error")
        contents = external.read_bytes()
        for root in (self.project.generation_log_path, self.project.evaluation_log_path):
            root.symlink_to(external.parent, target_is_directory=True)

        self.assertEqual(self.project.get_untracked_log_paths("generation"), [])
        self.assertEqual(self.project.get_untracked_log_paths("evaluation"), [])
        self.project.cleanup_incomplete_logs()

        self.assertTrue(self.project.generation_log_path.is_symlink())
        self.assertTrue(self.project.evaluation_log_path.is_symlink())
        self.assertEqual(external.read_bytes(), contents)

    def test_symlink_project_directory_is_not_followed(self):
        path = self.write_log(self.project.generation_log_path / "failure.json", "error")
        contents = path.read_bytes()
        actual = self.root / "moved"
        self.project.project_path.rename(actual)
        self.project.project_path.symlink_to(actual, target_is_directory=True)

        self.assertEqual(self.project.get_untracked_log_paths("generation"), [])
        self.project.cleanup_incomplete_logs()

        self.assertEqual(path.read_bytes(), contents)

    def test_cleanup_rechecks_a_file_replaced_by_a_symlink_during_read(self):
        candidate = self.write_log(self.project.generation_log_path / "failure.json", "error")
        external = self.write_log(self.root / "other-project" / "failure.json", "error")
        contents = external.read_bytes()

        def replace_with_symlink(location):
            log = read_eval_log(location)
            candidate.unlink()
            candidate.symlink_to(external)
            return log

        with patch(
            "evalsense.workflow.project.read_eval_log",
            side_effect=replace_with_symlink,
        ):
            self.project.cleanup_incomplete_logs()

        self.assertTrue(candidate.is_symlink())
        self.assertEqual(external.read_bytes(), contents)

    def test_cleanup_uses_the_discovered_path_instead_of_log_location(self):
        candidate = self.write_log(
            self.project.generation_log_path / "failure.json", "error"
        )
        external = self.write_log(self.root / "other-project" / "saved.json")
        contents = external.read_bytes()
        log = read_eval_log(str(candidate))
        log.location = str(external)

        with patch("evalsense.workflow.project.read_eval_log", return_value=log):
            self.project.cleanup_incomplete_logs()

        self.assertFalse(candidate.exists())
        self.assertEqual(external.read_bytes(), contents)
