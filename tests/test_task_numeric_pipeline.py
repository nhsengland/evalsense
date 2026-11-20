import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict
from inspect_ai.dataset import FieldSpec
from inspect_ai.log import EvalLog
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import Evaluator, ExperimentConfig
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.workflow import Pipeline, Project


@scorer(metrics=[accuracy()])
def numeric_accuracy() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        correct = math.isclose(
            float(state.output.completion),
            float(target.text),
            rel_tol=1e-6,
            abs_tol=0.0,
        )
        return Score(value=CORRECT if correct else INCORRECT)

    return score


def numeric_dataset(
    directory: str, values: list[float], *, string_targets: bool = False
) -> HuggingFaceDatasetManager:
    manager = HuggingFaceDatasetManager(
        "numeric-fixture", version="v1", splits=["test"], data_dir=directory
    )
    DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "id": [str(i) for i in range(len(values))],
                    "prediction": [str(value) for value in values],
                    "reference": (
                        [str(value) for value in values] if string_targets else values
                    ),
                    "measured": values,
                }
            )
        }
    ).save_to_disk(manager.main_data_path)
    return manager


def numeric_experiment(
    manager: HuggingFaceDatasetManager, mode: str, calls: list[str]
) -> ExperimentConfig:
    @solver
    def fixed_predictions(mode: str) -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            calls.append(str(state.sample_id))
            # Predictions come from an independent text column, never the target.
            assert isinstance(state.input, str)
            prediction = state.input if mode == "faithful" else "0.0"
            state.output = ModelOutput.from_content(model="fixture", content=prediction)
            return state

        return solve

    return ExperimentConfig(
        manager,
        GenerationSteps("fixed predictions", fixed_predictions(mode), version="1"),
        ModelConfig("mockllm/model"),
        field_spec=FieldSpec(
            input="prediction", target="reference", id="id", metadata=["measured"]
        ),
        evaluator=Evaluator("numeric accuracy", numeric_accuracy(), version="1"),
    )


def run_numeric_experiment(
    experiment: ExperimentConfig, project: Project, log_format: str = "json"
) -> EvalLog:
    Pipeline(experiment, project).run(
        show_progress=False,
        eval_kwargs={"display": "none", "log_format": log_format},
    )
    log = project.get_log(experiment.evaluation_record)
    assert log is not None
    return log


def reported_accuracy(log: EvalLog) -> float:
    assert log.results is not None
    return log.results.scores[0].metrics["accuracy"].value


class NumericPipelineTests(unittest.TestCase):
    def test_tiny_numeric_references_preserve_the_correct_model_ranking(self):
        self.check_accuracy(
            [1e-11, 2e-11, 4e-11, -1e-11, -2e-11, -4e-11], string_targets=False
        )

    def test_ordinary_and_high_precision_references_preserve_scores(self):
        self.check_accuracy(
            [0.125, 2.5, -3.75, 1.2345678901234567, 123.123456789012, -6.6743e-11],
            string_targets=False,
        )

    def test_equivalent_string_targets_preserve_the_same_scores(self):
        self.check_accuracy(
            [1e-11, 2e-11, 4e-11, -1e-11, -2e-11, -4e-11], string_targets=True
        )

    def check_accuracy(self, values: list[float], *, string_targets: bool):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp) / "projects"),
        ):
            manager = numeric_dataset(tmp, values, string_targets=string_targets)
            loaded = manager.load(retrieve=False)
            self.assertEqual([float(v) for v in loaded["reference"]], values)
            project = Project("numeric-accuracy")
            for mode in ("faithful", "zero"):
                with self.subTest(mode=mode):
                    predictions = values if mode == "faithful" else [0.0] * len(values)
                    expected = sum(
                        math.isclose(p, y, rel_tol=1e-6, abs_tol=0.0)
                        for p, y in zip(predictions, values, strict=True)
                    ) / len(values)
                    calls: list[str] = []
                    experiment = numeric_experiment(manager, mode, calls)
                    log = run_numeric_experiment(experiment, project)
                    self.assertEqual(log.status, "success")
                    self.assertEqual(reported_accuracy(log), expected)
                    self.assertEqual(expected, 1.0 if mode == "faithful" else 0.0)
                    self.assertEqual(len(calls), len(values))
                    assert log.samples is not None
                    samples = sorted(
                        log.samples, key=lambda sample: int(str(sample.id))
                    )
                    self.assertEqual(len(samples), len(values))
                    for sample, value, prediction in zip(
                        samples, values, predictions, strict=True
                    ):
                        assert isinstance(sample.target, str)
                        self.assertEqual(float(sample.target).hex(), value.hex())
                        assert sample.metadata is not None
                        self.assertEqual(sample.metadata["measured"].hex(), value.hex())
                        self.assertEqual(float(sample.output.completion), prediction)
                        assert sample.scores is not None
                        self.assertEqual(
                            next(iter(sample.scores.values())).value,
                            CORRECT if mode == "faithful" else INCORRECT,
                        )


if __name__ == "__main__":
    unittest.main()
