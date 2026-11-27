from unittest.mock import patch

from inspect_ai.log import read_eval_log

from evalsense.workflow import Project

from project_log_fixture import ProjectLogFixture


class PartialProjectLogTests(ProjectLogFixture):
    def test_partial_and_unrelated_files_survive_repeated_project_loading(self):
        contents = {}
        for root in (self.project.generation_log_path, self.project.evaluation_log_path):
            for directory in (root, root / "generation" / "attempt"):
                directory.mkdir(parents=True)
                for name, data in (
                    ("empty.json", b""),
                    ("partial.json", b'{"status":'),
                    ("partial.eval", b"PK"),
                    ("invalid.eval", b"not an Inspect archive"),
                    ("writing.tmp", b"unfinished"),
                    ("notes.txt", b"keep these notes"),
                ):
                    path = directory / name
                    path.write_bytes(data)
                    contents[path] = data
                (directory / "directory.json").mkdir()
        metadata = self.project.project_path / Project.METADATA_FILE
        contents[metadata] = metadata.read_bytes()

        for _ in range(3):
            reopened = Project(self.project.name)
            self.assertEqual(reopened.records, self.project.records)
            self.assertEqual(
                {path: path.read_bytes() for path in contents}, contents
            )

    def test_loading_does_not_attempt_to_read_even_unreadable_logs(self):
        path = self.write_log(self.project.generation_log_path / "saved.json")
        contents = path.read_bytes()

        with patch(
            "evalsense.workflow.project.read_eval_log",
            side_effect=PermissionError("log is temporarily unreadable"),
        ) as reader:
            reopened = Project(self.project.name)

        reader.assert_not_called()
        self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(reopened.records.generation, {})

    def test_an_open_writer_can_finish_a_partial_log_after_project_reopening(self):
        for suffix in (".json", ".eval"):
            with self.subTest(suffix=suffix):
                path = self.write_log(
                    self.project.generation_log_path / ("writing" + suffix)
                )
                contents = path.read_bytes()
                boundary = len(contents) // 2
                path.write_bytes(contents[:boundary])

                with path.open("ab") as writer:
                    reopened = Project(self.project.name)
                    self.assertEqual(path.read_bytes(), contents[:boundary])
                    writer.write(contents[boundary:])

                self.assertEqual(path.read_bytes(), contents)
                self.assertEqual(read_eval_log(str(path)).status, "success")
                self.assertEqual(reopened.records.generation, {})
