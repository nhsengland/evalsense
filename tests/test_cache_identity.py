from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Any, cast
from unittest.mock import Mock, patch

from datasets import Dataset
from inspect_ai.dataset import FieldSpec, Sample
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, write_eval_log
from inspect_ai.scorer import Score, exact, mean
from inspect_ai.solver import generate, system_message

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import (
    EvaluationRecord,
    Evaluator,
    ExperimentBatchConfig,
    ExperimentConfig,
    GenerationRecord,
    ResultRecord,
    TaskConfig,
)
from evalsense.evaluation.evaluators import (
    get_bertscore_evaluator,
    get_g_eval_evaluator,
    get_qags_evaluator,
)
from evalsense.evaluation.evaluators.qags import QagsConfig
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.tasks import DefaultTaskPreprocessor, TaskPreprocessor
from evalsense.utils.cache import CacheIdentityError, fingerprint
from evalsense.workflow import Pipeline, Project
from evalsense.workflow.project import ProjectRecords


def experiment(prompt="first", *, dataset=None, evaluator=None, **kwargs):
    return ExperimentConfig(
        dataset_manager=dataset
        or HuggingFaceDatasetManager("fixture", splits=["test"], version="v1"),
        generation_steps=GenerationSteps(
            "same name", [system_message(prompt), generate()]
        ),
        model_config=ModelConfig("mockllm/model"),
        evaluator=evaluator or Evaluator("exact", exact()),
        **kwargs,
    )


class ExampleQagsConfig(QagsConfig):
    def get_question_generation_prompt(self, **kwargs):
        return "Questions?"

    def get_answer_generation_prompt(self, **kwargs):
        return "yes"


class CacheIdentityTests(unittest.TestCase):
    def test_deterministic_configuration(self):
        self.assertEqual(
            fingerprint({"a": 1, "b": [2, 3]}), fingerprint({"b": [2, 3], "a": 1})
        )
        self.assertNotEqual(fingerprint([1, 2]), fingerprint([2, 1]))
        self.assertNotEqual(fingerprint({1: "x"}), fingerprint({"1": "x"}))
        self.assertNotEqual(fingerprint(1), fingerprint(True))
        self.assertNotEqual(fingerprint(1), fingerprint(1.0))

    def test_fingerprint_survives_process_restart(self):
        code = "from evalsense.utils.cache import fingerprint; print(fingerprint({'a':1,'b':[2,3]}))"
        output = subprocess.check_output(
            [sys.executable, "-c", code], text=True
        ).strip()
        self.assertEqual(output, fingerprint({"b": [2, 3], "a": 1}))

    def test_unknown_configuration_and_nonfinite_values_are_rejected(self):
        for value in (object(), float("nan"), float("inf")):
            with self.subTest(value=type(value)), self.assertRaises(CacheIdentityError):
                fingerprint(value)
        for version in ("", " ", 1):
            with self.subTest(version=version), self.assertRaises(CacheIdentityError):
                fingerprint({}, version=cast(Any, version))

    def test_custom_solver_requires_version(self):
        async def solve(state, generate):
            return state

        steps = GenerationSteps("custom", solve)
        with self.assertRaisesRegex(CacheIdentityError, "version"):
            _ = steps.cache_key
        steps.version = "v1"
        first = steps.cache_key
        self.assertEqual(first, steps.cache_key)
        steps.version = "v2"
        self.assertNotEqual(first, steps.cache_key)

    def test_prompt_change_splits_generation_keys(self):
        first, second = experiment("one"), experiment("two")
        self.assertNotEqual(first.generation_record, second.generation_record)
        self.assertNotEqual(first.evaluation_record, second.evaluation_record)
        self.assertEqual(first.generation_record, experiment("one").generation_record)
        self.assertEqual(
            len(Pipeline([first, second], Mock(spec=Project)).generation_experiments), 2
        )

    def test_solver_order_and_generation_parameters_are_part_of_identity(self):
        first = experiment()
        second = replace(
            first,
            generation_steps=GenerationSteps(
                "same name", [generate(), system_message("first")]
            ),
        )
        self.assertNotEqual(first.generation_record, second.generation_record)
        second = replace(
            first,
            generation_steps=GenerationSteps(
                "same name", [system_message("first"), generate(temperature=1)]
            ),
        )
        self.assertNotEqual(first.generation_record, second.generation_record)

    def test_field_mapping_changes_generation_but_not_preprocessing_definition(self):
        first = experiment(field_spec=FieldSpec(input="one"))
        second = experiment(field_spec=FieldSpec(input="two"))
        self.assertNotEqual(first.generation_record, second.generation_record)
        self.assertEqual(
            first.task_preprocessor.cache_key, second.task_preprocessor.cache_key
        )
        self.assertEqual(
            experiment().generation_record,
            experiment(field_spec=FieldSpec()).generation_record,
        )

    def test_custom_field_mapper_requires_version(self):
        def mapper(row):
            return Sample(input=row["input"], target=row["target"])

        first = experiment(field_spec=mapper)
        with self.assertRaisesRegex(CacheIdentityError, "version"):
            _ = first.generation_record
        first.field_spec_version = "1"
        second = replace(first, field_spec_version="2")
        self.assertNotEqual(first.generation_record, second.generation_record)
        task = TaskConfig(
            first.dataset_manager,
            first.generation_steps,
            field_spec=mapper,
            field_spec_version="1",
        )
        assert first.evaluator is not None
        batch = ExperimentBatchConfig([task], [first.model_config], [first.evaluator])
        self.assertEqual(
            batch.all_experiments[0].generation_record, first.generation_record
        )

    def test_preprocessing_version_changes_generation_and_task_files(self):
        def preprocess(hf_dataset, dataset_manager):
            return hf_dataset

        first = experiment(
            task_preprocessor=TaskPreprocessor("same", preprocess, version="1")
        )
        second = experiment(
            task_preprocessor=TaskPreprocessor("same", preprocess, version="2")
        )
        self.assertNotEqual(first.generation_record, second.generation_record)
        with self.assertRaisesRegex(CacheIdentityError, "version"):
            _ = TaskPreprocessor("custom", preprocess).cache_key

    def test_changed_g_eval_configuration_only_invalidates_evaluation(self):
        base: dict[str, Any] = dict(
            model_config=ModelConfig("mockllm/judge"),
            prompt_template="Score {prediction}.",
        )
        baseline = experiment(evaluator=get_g_eval_evaluator(**base))
        for field, value in [
            ("prompt_template", "Assess {prediction}."),
            ("min_score", 0),
            ("max_score", 5),
            ("logprobs", False),
            ("top_logprobs", 5),
            ("normalise", False),
            ("metrics", [mean()]),
        ]:
            with self.subTest(field=field):
                other = experiment(
                    evaluator=get_g_eval_evaluator(**(base | {field: value}))
                )
                self.assertEqual(baseline.generation_record, other.generation_record)
                self.assertNotEqual(baseline.evaluation_record, other.evaluation_record)
        other = experiment(
            evaluator=get_g_eval_evaluator(
                **(
                    base
                    | {
                        "model_config": ModelConfig(
                            "mockllm/judge", generation_args={"temperature": 1.0}
                        )
                    }
                )
            )
        )
        self.assertNotEqual(baseline.evaluation_record, other.evaluation_record)

    def test_bertscore_parameters_have_distinct_keys_without_loading_models(self):
        with patch(
            "evalsense.evaluation.evaluators.bertscore.evaluate.load",
            return_value=Mock(),
        ):
            baseline = get_bertscore_evaluator()
            for params in [
                {"batch_size": 1},
                {"lang": "fr"},
                {"num_layers": 4},
                {"rescale_with_baseline": True},
                {"use_fast_tokenizer": True},
                {"model_type": "another/model"},
            ]:
                with self.subTest(params=params):
                    self.assertNotEqual(
                        baseline.cache_key,
                        get_bertscore_evaluator(
                            **cast(dict[str, Any], params)
                        ).cache_key,
                    )

    def test_qags_configuration_and_prompt_version_are_included(self):
        config = ExampleQagsConfig(answer_comparison_mode="ternary")
        options: dict[str, Any] = dict(
            config=config, model_config=ModelConfig("mockllm/judge")
        )
        with self.assertRaisesRegex(CacheIdentityError, "version"):
            _ = get_qags_evaluator(**options).cache_key
        first = get_qags_evaluator(**options, version="1")
        before = first.cache_key
        config.answer_comparison_mode = "exact"
        self.assertNotEqual(before, first.cache_key)
        self.assertNotEqual(
            first.cache_key, get_qags_evaluator(**options, version="2").cache_key
        )

    def test_custom_evaluator_requires_version_or_declared_parameters(self):
        async def score(state, target):
            return Score(value=1)

        evaluator = Evaluator("custom", score)
        with self.assertRaisesRegex(CacheIdentityError, "version"):
            _ = evaluator.cache_key
        evaluator.version = "1"
        first = evaluator.cache_key
        evaluator.version = "2"
        self.assertNotEqual(first, evaluator.cache_key)

    def test_declared_parameters_do_not_hide_custom_scorer_implementation(self):
        async def score(state, target):
            return Score(value=1)

        evaluator = Evaluator("custom", score, cache_parameters={"threshold": 0.5})
        with self.assertRaisesRegex(CacheIdentityError, "version"):
            _ = evaluator.cache_key
        evaluator.version = "1"
        first = evaluator.cache_key
        evaluator.cache_parameters = {"threshold": 0.9}
        self.assertNotEqual(first, evaluator.cache_key)

    def test_record_equality_hash_order_and_roundtrip(self):
        first = experiment().evaluation_record
        variants = [
            first,
            first.model_copy(update={"generator_fingerprint": "g"}),
            first.model_copy(update={"task_fingerprint": "t"}),
            first.model_copy(update={"evaluator_fingerprint": "e"}),
        ]
        self.assertEqual(len(set(variants)), 4)
        self.assertEqual(sorted(sorted(variants, reverse=True)), sorted(variants))
        for item in variants:
            self.assertEqual(
                item, EvaluationRecord.model_validate_json(item.model_dump_json())
            )
            self.assertEqual(
                item.generation_record.get_evaluation_record(
                    item.evaluator_name, item.evaluator_fingerprint
                ),
                item,
            )
        records = ProjectRecords()
        records.evaluation = {key: ResultRecord(status="success") for key in variants}
        restored = ProjectRecords.model_validate_json(records.model_dump_json())
        self.assertEqual(
            restored.evaluation,
            records.evaluation,
        )

    def test_legacy_records_are_readable_but_not_reused(self):
        current = experiment().generation_record
        legacy = GenerationRecord.model_validate(
            current.model_dump(exclude={"generator_fingerprint", "task_fingerprint"})
        )
        self.assertNotEqual(legacy, current)
        self.assertEqual(legacy.generator_fingerprint, "")
        self.assertEqual(
            legacy, GenerationRecord.model_validate_json(legacy.model_dump_json())
        )

    def test_meta_grouping_ignores_only_the_tier_generator(self):
        first = experiment("tier one").evaluation_record
        second = experiment("tier two").evaluation_record
        self.assertNotEqual(first, second)
        self.assertEqual(
            first.get_meta_grouped_record("quality"),
            second.get_meta_grouped_record("quality"),
        )
        changed = second.model_copy(update={"evaluator_fingerprint": "changed"})
        self.assertNotEqual(
            first.get_meta_grouped_record("quality"),
            changed.get_meta_grouped_record("quality"),
        )

    def test_task_cache_separates_splits_content_and_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def preprocessing(hf_dataset, dataset_manager):
                calls.append(dataset_manager.splits)
                return hf_dataset

            pre = TaskPreprocessor("custom", preprocessing, version="1")
            train = HuggingFaceDatasetManager(
                "fixture", version="v1", splits=["train"], data_dir=tmp
            )
            test = HuggingFaceDatasetManager(
                "fixture", version="v1", splits=["test"], data_dir=tmp
            )
            one = Dataset.from_dict({"input": ["train"], "target": ["yes"]})
            two = Dataset.from_dict({"input": ["test"], "target": ["no"]})
            self.assertEqual(pre(one, train)[0].input, "train")
            self.assertEqual(pre(one, train)[0].input, "train")
            self.assertEqual(len(calls), 1)
            self.assertEqual(pre(two, test)[0].input, "test")
            self.assertEqual(pre(two, train)[0].input, "test")
            pre.version = "2"
            self.assertEqual(pre(two, train)[0].input, "test")
            self.assertEqual(len(calls), 4)
            self.assertEqual(
                len(list((train.version_path / "tasks").glob("*.jsonl"))), 4
            )

    def test_existing_name_only_task_cache_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = HuggingFaceDatasetManager(
                "fixture", version="v1", splits=["test"], data_dir=tmp
            )
            tasks = manager.version_path / "tasks"
            tasks.mkdir(parents=True)
            old = tasks / "fixture-Identity.jsonl"
            old.write_text(json.dumps({"input": "stale", "target": "no"}) + "\n")
            fresh = Dataset.from_dict({"input": ["fresh"], "target": ["yes"]})
            self.assertEqual(
                DefaultTaskPreprocessor()(fresh, manager)[0].input, "fresh"
            )
            self.assertTrue(old.exists())

    def test_evaluator_configurations_use_different_physical_logs(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp)),
        ):
            project = Project("test")
            project.generation_log_path.mkdir(parents=True)
            path = project.generation_log_path / "2026_task_run.json"
            data = EvalLog(
                status="success",
                eval=EvalSpec(
                    created="2026-01-01T00:00:00Z",
                    task="task",
                    model="model",
                    dataset=EvalDataset(),
                    config=EvalConfig(),
                ),
                samples=[],
            )
            write_eval_log(data, location=str(path))
            first = experiment(
                evaluator=get_g_eval_evaluator(
                    model_config=ModelConfig("mockllm/judge"),
                    prompt_template="one {prediction}",
                )
            )
            second = experiment(
                evaluator=get_g_eval_evaluator(
                    model_config=ModelConfig("mockllm/judge"),
                    prompt_template="two {prediction}",
                )
            )
            project.update_record(
                first.generation_record,
                ResultRecord(status="success", log_location=str(path)),
            )
            a = project.get_record(
                first.evaluation_record, init_eval_record_from_generations=True
            )
            b = project.get_record(
                second.evaluation_record, init_eval_record_from_generations=True
            )
            assert a is not None and b is not None and a.log_location and b.log_location
            self.assertNotEqual(a.log_location, b.log_location)
            self.assertTrue(Path(a.log_location).exists())
            self.assertTrue(Path(b.log_location).exists())
            renamed = first.evaluation_record.model_copy(
                update={"evaluator_name": "another name"}
            )
            c = project.get_record(renamed, init_eval_record_from_generations=True)
            assert c is not None and c.log_location
            self.assertNotEqual(c.log_location, a.log_location)
            self.assertTrue(Path(c.log_location).exists())
            self.assertEqual(len(Project("test").records.evaluation), 3)


if __name__ == "__main__":
    unittest.main()
