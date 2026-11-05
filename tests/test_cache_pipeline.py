from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, Scorer, Target, exact, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver, system_message

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import Evaluator, ExperimentConfig
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.utils.cache import CacheIdentityError
from evalsense.workflow import Pipeline, Project


@scorer(metrics=[mean()])
def fixed_score(value: float) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=value)

    return score


class PipelineCacheTests(unittest.TestCase):
    def test_changed_prompts_generate_new_outputs_and_identical_runs_reuse(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp) / "projects"),
        ):
            calls = []

            @solver
            def from_system_prompt() -> Solver:
                async def solve(state: TaskState, generate: Generate) -> TaskState:
                    text = state.messages[0].text
                    calls.append(text)
                    state.output = ModelOutput.from_content(
                        model="fixture", content=text
                    )
                    return state

                return solve

            manager = HuggingFaceDatasetManager(
                "fixture", version="v1", splits=["test"], data_dir=tmp
            )
            DatasetDict(
                {
                    "test": Dataset.from_dict(
                        {"id": ["sample"], "input": ["input"], "target": ["yes"]}
                    )
                }
            ).save_to_disk(manager.main_data_path)
            project = Project("run")

            def config(prompt):
                return ExperimentConfig(
                    manager,
                    GenerationSteps(
                        "same",
                        [system_message(prompt), from_system_prompt()],
                        version="1",
                    ),
                    ModelConfig("mockllm/model"),
                    evaluator=Evaluator("exact", exact()),
                )

            def run(exp):
                Pipeline(exp, project).run(
                    show_progress=False,
                    eval_kwargs={"display": "none", "log_format": "json"},
                )

            first = config("yes")
            run(first)
            run(config("yes"))
            self.assertEqual(calls, ["yes"])
            second = config("no")
            run(second)
            self.assertEqual(calls, ["yes", "no"])
            self.assertEqual(len(project.records.generation), 2)
            first_log = project.get_log(first.evaluation_record)
            second_log = project.get_log(second.evaluation_record)
            assert (
                first_log is not None
                and first_log.samples
                and second_log is not None
                and second_log.samples
            )
            self.assertEqual(first_log.samples[0].output.completion, "yes")
            self.assertEqual(second_log.samples[0].output.completion, "no")
            assert first_log.samples[0].scores and second_log.samples[0].scores
            self.assertEqual(
                next(iter(first_log.samples[0].scores.values())).value, "C"
            )
            self.assertEqual(
                next(iter(second_log.samples[0].scores.values())).value, "I"
            )

            # Change only the scorer parameters. Its generation must stay reusable,
            # while the evaluator name can remain unchanged.
            for value in (0.1, 0.9):
                exp = config("yes")
                exp.evaluator = Evaluator("fixed", fixed_score(value), version="1")
                run(exp)
                result = project.get_log(exp.evaluation_record)
                assert (
                    result is not None and result.samples and result.samples[0].scores
                )
                self.assertEqual(
                    next(iter(result.samples[0].scores.values())).value, value
                )
            self.assertEqual(calls, ["yes", "no"])
            self.assertEqual(len(project.records.evaluation), 4)
            self.assertEqual(len(Project("run").records.evaluation), 4)

    def test_nested_custom_solvers_need_versions(self):
        from inspect_ai.solver import chain

        async def custom(state: TaskState, generate: Generate) -> TaskState:
            return state

        steps = GenerationSteps("nested", chain(system_message("test"), custom))
        with self.assertRaises(CacheIdentityError):
            _ = steps.cache_key
        steps.version = "1"
        self.assertEqual(steps.cache_key, steps.cache_key)


if __name__ == "__main__":
    unittest.main()
