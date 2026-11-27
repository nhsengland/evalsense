from inspect_ai.log import EvalMetric, EvalResults, EvalScore, read_eval_log, write_eval_log

from evalsense.evaluation import RecordStatus, ResultRecord
from evalsense.workflow import Project

from project_log_fixture import ProjectLogFixture


class OrphanedEvaluationLogTests(ProjectLogFixture):
    def scored_log(self, name, value):
        path = self.write_log(self.project.evaluation_log_path / name, task="score")
        log = read_eval_log(str(path))
        log.results = EvalResults(
            total_samples=4,
            completed_samples=4,
            scores=[
                EvalScore(
                    name="accuracy",
                    scorer="accuracy",
                    metrics={"mean": EvalMetric(name="mean", value=value)},
                )
            ],
        )
        write_eval_log(log, location=str(path))
        return path

    def test_evaluation_logs_in_every_status_survive_loading_in_both_formats(self):
        statuses: tuple[RecordStatus, ...] = (
            "started", "success", "error", "cancelled"
        )
        contents = {}
        for suffix in (".json", ".eval"):
            for status in statuses:
                path = self.write_log(
                    self.project.evaluation_log_path / (status + suffix),
                    status,
                    task="score",
                )
                contents[path] = path.read_bytes()
        metadata = self.project.project_path / Project.METADATA_FILE
        contents[metadata] = metadata.read_bytes()

        reopened = Project(self.project.name)

        self.assertEqual({path: path.read_bytes() for path in contents}, contents)
        self.assertEqual(reopened.get_logs("evaluation"), {})
        self.assertEqual(reopened.get_logs("evaluation", status="success"), {})

    def test_orphaned_aggregate_scores_remain_readable_after_reopening(self):
        for suffix in (".json", ".eval"):
            with self.subTest(suffix=suffix):
                path = self.scored_log("unregistered" + suffix, 0.75)
                contents = path.read_bytes()

                reopened = Project(self.project.name)
                result = read_eval_log(str(path)).results

                self.assertEqual(path.read_bytes(), contents)
                assert result is not None
                self.assertEqual(result.scores[0].metrics["mean"].value, 0.75)
                self.assertIn(path, reopened.get_untracked_log_paths("evaluation"))
                self.assertEqual(reopened.get_logs("evaluation"), {})

    def test_orphaned_scores_do_not_replace_registered_results(self):
        registered = self.scored_log("registered.json", 0.25)
        orphaned = self.scored_log("unregistered.eval", 0.75)
        record = ResultRecord(status="success", log_location=str(registered))
        self.project.update_record(self.experiment.evaluation_record, record)
        contents = orphaned.read_bytes()

        reopened = Project(self.project.name)
        logs = reopened.get_logs("evaluation", status="success")

        self.assertEqual(set(logs), {self.experiment.evaluation_record})
        result = logs[self.experiment.evaluation_record].results
        assert result is not None
        self.assertEqual(result.scores[0].metrics["mean"].value, 0.25)
        self.assertEqual(reopened.get_record(self.experiment.evaluation_record), record)
        self.assertEqual(orphaned.read_bytes(), contents)

    def test_preserved_evaluation_can_be_registered_explicitly(self):
        path = self.scored_log("recoverable.json", 0.75)
        contents = path.read_bytes()
        reopened = Project(self.project.name)
        self.assertEqual(reopened.get_logs("evaluation"), {})

        reopened.update_record(
            self.experiment.evaluation_record,
            ResultRecord(status="success", log_location=str(path)),
        )
        saved = Project(self.project.name)

        self.assertEqual(saved.get_untracked_log_paths("evaluation"), [])
        log = saved.get_log(self.experiment.evaluation_record)
        assert log is not None and log.results is not None
        self.assertEqual(log.results.scores[0].metrics["mean"].value, 0.75)
        self.assertEqual(path.read_bytes(), contents)
