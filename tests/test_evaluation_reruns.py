"""Evaluation reruns using saved logs, real Inspect scoring and no model calls."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import Evaluator, ExperimentConfig, GenerationSource
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.workflow import Pipeline, Project
from evalsense.workflow.analysers import TabularResultAnalyser


class RerunFixture:
    def __init__(self, storage: Path, outputs: list[str]):
        self.generated: list[str] = []
        self.scored: list[str] = []
        self.fail_scoring = False
        remaining = iter(outputs)

        @solver
        def sequenced_output() -> Solver:
            async def solve(state: TaskState, generate: Generate) -> TaskState:
                output = next(remaining)
                self.generated.append(output)
                state.output = ModelOutput.from_content(model="fixture", content=output)
                return state

            return solve

        @scorer(metrics=[mean()])
        def tracked_accuracy() -> Scorer:
            async def score(state: TaskState, target: Target) -> Score:
                output = state.output.completion
                self.scored.append(output)
                if self.fail_scoring:
                    raise ValueError("fixture scoring failure")
                return Score(value=float(output == target.text))

            return score

        manager = HuggingFaceDatasetManager(
            "fixture", version="1", splits=["test"], data_dir=str(storage)
        )
        DatasetDict(
            {
                "test": Dataset.from_dict(
                    {"id": ["sample"], "input": ["input"], "target": ["yes"]}
                )
            }
        ).save_to_disk(manager.main_data_path)
        self.experiment = ExperimentConfig(
            manager,
            GenerationSteps("sequence", sequenced_output(), version="1"),
            ModelConfig("mockllm/model"),
            evaluator=Evaluator("accuracy", tracked_accuracy(), version="1"),
        )
        self.project = Project("rerun")

    @property
    def pipeline(self) -> Pipeline:
        return Pipeline(self.experiment, self.project)

    def reopen(self) -> None:
        self.project = Project("rerun")

    def run(self, log_format: str, force: bool = False) -> None:
        self.pipeline.run(
            show_progress=False,
            force_rerun=force,
            eval_kwargs={"display": "none", "log_format": log_format},
        )

    def generate(self, log_format: str, force: bool = False) -> None:
        self.pipeline.generate(
            show_progress=False,
            force_rerun=force,
            eval_kwargs={"display": "none", "log_format": log_format},
        )

    def evaluate(self, force: bool = False) -> None:
        self.pipeline.evaluate(
            show_progress=False,
            force_rerun=force,
        )

    def generation_path(self) -> Path:
        record = self.project.get_record(self.experiment.generation_record)
        assert record is not None and record.log_location
        return Path(record.log_location)


class EvaluationRerunTests(unittest.TestCase):
    @contextmanager
    def fixture(self, outputs: list[str]) -> Iterator[RerunFixture]:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp) / "projects"),
        ):
            yield RerunFixture(Path(tmp), outputs)

    def assert_result(
        self, fixture: RerunFixture, output: str, score: float
    ) -> GenerationSource:
        project = fixture.project
        experiment = fixture.experiment
        generation = project.get_record(experiment.generation_record)
        evaluation = project.get_record(experiment.evaluation_record)
        assert generation is not None and generation.log_location
        assert evaluation is not None
        self.assertEqual(generation.status, "success")
        self.assertEqual(evaluation.status, "success")
        source = GenerationSource.capture(generation.log_location)
        self.assertEqual(evaluation.source_generation, source)

        generation_log = project.get_log(experiment.generation_record)
        evaluation_log = project.get_log(experiment.evaluation_record)
        assert generation_log is not None and generation_log.samples
        assert evaluation_log is not None and evaluation_log.samples
        self.assertEqual(generation_log.eval.run_id, evaluation_log.eval.run_id)
        self.assertEqual(generation_log.eval.task_id, evaluation_log.eval.task_id)
        self.assertEqual(generation_log.eval.created, evaluation_log.eval.created)
        self.assertEqual(generation_log.samples[0].output.completion, output)
        self.assertEqual(evaluation_log.samples[0].output.completion, output)
        sample_scores = evaluation_log.samples[0].scores
        assert sample_scores is not None
        self.assertEqual(next(iter(sample_scores.values())).value, score)
        assert evaluation_log.results is not None
        self.assertEqual(
            evaluation_log.results.scores[0].metrics["mean"].value, score
        )
        summary = TabularResultAnalyser(output_format="polars")(project)
        self.assertEqual(summary["tracked_accuracy/mean"].to_list(), [score])
        return source

    def test_cached_evaluation_and_forced_rescore_keep_generation_unchanged(self):
        for log_format in ("json", "eval"):
            with self.subTest(log_format=log_format), self.fixture(["yes"]) as fixture:
                fixture.run(log_format)
                source = self.assert_result(fixture, "yes", 1.0)
                generation_path = fixture.generation_path()
                generation_bytes = generation_path.read_bytes()

                fixture.evaluate()
                fixture.reopen()
                fixture.evaluate()
                self.assertEqual(fixture.scored, ["yes"])
                self.assertEqual(self.assert_result(fixture, "yes", 1.0), source)

                fixture.evaluate(force=True)
                self.assertEqual(fixture.scored, ["yes", "yes"])
                self.assertEqual(fixture.generated, ["yes"])
                self.assertEqual(self.assert_result(fixture, "yes", 1.0), source)
                self.assertEqual(fixture.generation_path(), generation_path)
                self.assertEqual(generation_path.read_bytes(), generation_bytes)

    def test_failed_evaluation_keeps_replacement_source_for_retry_after_reopen(self):
        for log_format in ("json", "eval"):
            with (
                self.subTest(log_format=log_format),
                self.fixture(["yes", "no"]) as fixture,
            ):
                fixture.run(log_format)
                previous_source = self.assert_result(fixture, "yes", 1.0)
                fixture.generate(log_format, force=True)
                generation_path = fixture.generation_path()
                generation_bytes = generation_path.read_bytes()
                source = GenerationSource.capture(generation_path)
                self.assertNotEqual(source, previous_source)

                fixture.fail_scoring = True
                fixture.evaluate(force=True)
                fixture.reopen()
                failed = fixture.project.get_record(fixture.experiment.evaluation_record)
                assert failed is not None
                self.assertEqual(failed.status, "error")
                self.assertTrue(failed.error_message)
                self.assertEqual(failed.source_generation, source)
                self.assertEqual(fixture.scored, ["yes", "no"])

                fixture.fail_scoring = False
                fixture.evaluate()
                fixture.reopen()
                self.assertEqual(self.assert_result(fixture, "no", 0.0), source)
                self.assertEqual(fixture.scored, ["yes", "no", "no"])
                self.assertEqual(fixture.generated, ["yes", "no"])
                self.assertEqual(fixture.generation_path(), generation_path)
                self.assertEqual(generation_path.read_bytes(), generation_bytes)


    def assert_forced_sequence(self, first: str, second: str) -> None:
        for log_format in ("json", "eval"):
            with (
                self.subTest(log_format=log_format),
                self.fixture([first, second]) as fixture,
            ):
                generation_key = fixture.experiment.generation_record
                evaluation_key = fixture.experiment.evaluation_record
                fixture.run(log_format)
                previous_source = self.assert_result(
                    fixture, first, float(first == "yes")
                )
                fixture.reopen()
                fixture.run(log_format, force=True)
                source = self.assert_result(fixture, second, float(second == "yes"))
                self.assertNotEqual(source, previous_source)
                self.assertNotEqual(source.log_location, previous_source.log_location)
                self.assertEqual(fixture.generation_path().suffix, "." + log_format)
                self.assertEqual(fixture.experiment.generation_record, generation_key)
                self.assertEqual(fixture.experiment.evaluation_record, evaluation_key)
                self.assertEqual(fixture.generated, [first, second])
                self.assertEqual(fixture.scored, [first, second])

                fixture.reopen()
                self.assertEqual(
                    self.assert_result(fixture, second, float(second == "yes")), source
                )

    def test_forced_rerun_reports_deterioration_after_reopen(self):
        self.assert_forced_sequence("yes", "no")

    def test_forced_rerun_reports_improvement_after_reopen(self):
        self.assert_forced_sequence("no", "yes")

    def test_identical_outputs_still_record_the_new_generation_source(self):
        self.assert_forced_sequence("yes", "yes")

    def test_regeneration_refreshes_evaluation_without_forcing_scoring(self):
        for log_format in ("json", "eval"):
            with (
                self.subTest(log_format=log_format),
                self.fixture(["yes", "no"]) as fixture,
            ):
                fixture.run(log_format)
                previous_source = self.assert_result(fixture, "yes", 1.0)
                fixture.generate(log_format, force=True)
                generation_path = fixture.generation_path()
                generation_bytes = generation_path.read_bytes()
                source = GenerationSource.capture(generation_path)
                self.assertNotEqual(source, previous_source)

                fixture.reopen()
                fixture.evaluate()
                self.assertEqual(self.assert_result(fixture, "no", 0.0), source)
                self.assertEqual(fixture.generated, ["yes", "no"])
                self.assertEqual(fixture.scored, ["yes", "no"])
                self.assertEqual(generation_path.read_bytes(), generation_bytes)
                fixture.reopen()
                self.assertEqual(self.assert_result(fixture, "no", 0.0), source)


if __name__ == "__main__":
    unittest.main()
