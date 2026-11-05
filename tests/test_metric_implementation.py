"""Metric revision, dependency identity, and persisted result regressions."""

import asyncio
from dataclasses import FrozenInstanceError, replace
from importlib.metadata import PackageNotFoundError
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from typing import Any, cast
import unittest
from unittest.mock import Mock, patch

from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalSample,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import SampleScore, Score, Scorer, Target
from inspect_ai.solver import TaskState, generate

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import (
    Evaluator,
    EvaluationRecord,
    ExperimentConfig,
    ResultRecord,
)
from evalsense.evaluation.evaluators import (
    BertScoreCalculator,
    BleuPrecisionScoreCalculator,
    RougeScoreCalculator,
    bleu_metric,
    get_bertscore_evaluator,
    get_bleu_evaluator,
    get_rouge_evaluator,
)
from evalsense.evaluation.metric_implementation import MetricImplementation
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.workflow import Pipeline, Project
from evalsense.workflow.analysers import (
    MetaResultAnalyser,
    MetricCorrelationAnalyser,
    TabularResultAnalyser,
)
from evalsense.workflow.analysers._provenance import check_metric_implementations

MODULE = "evalsense.evaluation.metric_implementation"


def fake_metric(path, **kwargs):
    name = path.rsplit("/", 1)[-1]
    result = {
        "bleu": {"precisions": [0.75], "bleu": 0.25},
        "rouge": {"rouge1": 0.7, "rouge2": 0.5, "rougeL": 0.6},
        "bertscore": {
            "precision": [0.8],
            "recall": [0.6],
            "f1": [0.7],
            "hashcode": "fixture",
        },
    }[name]
    module = Mock()
    module._hash = MetricImplementation.resolve(name).source_hash
    module.compute.return_value = result
    return module


def experiment(evaluator):
    return ExperimentConfig(
        HuggingFaceDatasetManager("fixture", version="1", splits=["test"]),
        GenerationSteps("generate", generate()),
        ModelConfig("mockllm/model"),
        evaluator=evaluator,
    )


def source_log():
    return EvalLog(
        status="success",
        eval=EvalSpec(
            created="2026-01-01T00:00:00Z",
            task="fixture",
            model="mockllm/model",
            dataset=EvalDataset(name="fixture"),
            config=EvalConfig(),
        ),
        samples=[
            EvalSample(
                id=1,
                epoch=1,
                input="source",
                target="reference",
                output=ModelOutput.from_content(model="fixture", content="prediction"),
            )
        ],
    )


class ImplementationTests(unittest.TestCase):
    def test_all_loads_request_full_revisions_and_verify_content(self):
        with patch(f"{MODULE}.evaluate.load", side_effect=fake_metric) as load:
            for name in ("bleu", "rouge", "bertscore"):
                with self.subTest(metric=name):
                    implementation = MetricImplementation.resolve(name)
                    self.assertRegex(implementation.revision, r"^[0-9a-f]{40}$")
                    self.assertRegex(implementation.source_hash, r"^[0-9a-f]{64}$")
                    implementation.load()
                    load.assert_called_with(
                        f"evaluate-metric/{name}",
                        revision=implementation.revision,
                        module_type="metric",
                    )

    def test_wrong_or_missing_cached_hash_fails_instead_of_scoring(self):
        for value in (None, "other", "0" * 64):
            with (
                self.subTest(hash=value),
                patch(
                    f"{MODULE}.evaluate.load", return_value=SimpleNamespace(_hash=value)
                ),
            ):
                with self.assertRaisesRegex(ValueError, "pinned revision"):
                    MetricImplementation.resolve("bleu").load()

    def test_loading_errors_are_not_silently_retried_unpinned(self):
        with patch(f"{MODULE}.evaluate.load", side_effect=OSError("offline")) as load:
            with self.assertRaises(OSError):
                MetricImplementation.resolve("bleu").load()
            self.assertEqual(load.call_count, 1)

    def test_environment_snapshot_is_immutable_and_serializable(self):
        with (
            patch(f"{MODULE}.version", return_value="1.2.3"),
            patch(f"{MODULE}.platform.python_version", return_value="3.12.1"),
        ):
            implementation = MetricImplementation.resolve("bertscore")
            self.assertEqual(implementation, MetricImplementation.resolve("bertscore"))
        data = implementation.provenance
        self.assertEqual(data["packages"]["transformers"], "1.2.3")
        data["packages"]["transformers"] = "changed"
        self.assertEqual(implementation.provenance["packages"]["transformers"], "1.2.3")
        with self.assertRaises(FrozenInstanceError):
            cast(Any, implementation).revision = "changed"
        self.assertEqual(
            json.loads(implementation.provenance_json), implementation.provenance
        )

    def test_missing_optional_versions_are_explicit(self):
        with patch(f"{MODULE}.version", side_effect=PackageNotFoundError):
            implementation = MetricImplementation.resolve("bertscore")
        self.assertIsNone(implementation.provenance["packages"]["torch"])

    def test_changed_dependency_or_revision_changes_evaluation_not_generation(self):
        with (
            patch(f"{MODULE}.version", return_value="1"),
            patch(f"{MODULE}.evaluate.load", side_effect=fake_metric),
        ):
            first = get_rouge_evaluator()
        with (
            patch(f"{MODULE}.version", return_value="2"),
            patch(f"{MODULE}.evaluate.load", side_effect=fake_metric),
        ):
            second = get_rouge_evaluator()
        a, b = experiment(first), experiment(second)
        self.assertEqual(first.name, second.name)
        self.assertEqual(a.generation_record, b.generation_record)
        self.assertNotEqual(a.evaluation_record, b.evaluation_record)
        assert first.implementation is not None
        third = replace(
            first, implementation=replace(first.implementation, revision="a" * 40)
        )
        self.assertNotEqual(a.evaluation_record, experiment(third).evaluation_record)
        pipeline = Pipeline([a, b], Mock(spec=Project))
        self.assertEqual(len(pipeline.generation_experiments), 1)
        self.assertEqual(len(pipeline.evaluation_experiments), 2)

    def test_record_equality_order_roundtrip_and_meta_groups_keep_provenance(self):
        with patch(f"{MODULE}.evaluate.load", side_effect=fake_metric):
            record = experiment(get_bleu_evaluator()).evaluation_record
        legacy = EvaluationRecord.model_validate(
            record.model_dump(exclude={"evaluator_provenance_json"})
        )
        changed = record.model_copy(
            update={"evaluator_provenance_json": "another implementation"}
        )
        self.assertEqual(len(set([record, legacy, changed])), 3)
        self.assertEqual(
            sorted([record, changed, legacy]), sorted([legacy, record, changed])
        )
        self.assertEqual(
            record, EvaluationRecord.model_validate_json(record.model_dump_json())
        )
        self.assertEqual(record.generation_record, legacy.generation_record)
        self.assertNotEqual(
            record.get_meta_grouped_record("quality"),
            changed.get_meta_grouped_record("quality"),
        )
        self.assertEqual(legacy.evaluator_provenance_json, "")

    def test_custom_evaluators_keep_their_existing_identity(self):
        evaluator = Evaluator("custom", cast(Any, lambda: None), version="1")
        record = experiment(evaluator).evaluation_record
        self.assertEqual(record.evaluator_provenance_json, "")
        self.assertEqual(
            record,
            record.generation_record.get_evaluation_record(
                "custom", evaluator_fingerprint=record.evaluator_fingerprint
            ),
        )

    def test_direct_scores_preserve_values_and_include_manifest(self):
        cases = [
            (BleuPrecisionScoreCalculator, 0.75),
            (RougeScoreCalculator, {"ROUGE-1": 0.7, "ROUGE-2": 0.5, "ROUGE-L": 0.6}),
            (
                BertScoreCalculator,
                {
                    "BERTScore Precision": 0.8,
                    "BERTScore Recall": 0.6,
                    "BERTScore F1": 0.7,
                },
            ),
        ]
        for cls, expected in cases:
            with (
                self.subTest(calculator=cls.__name__),
                patch(f"{MODULE}.evaluate.load", side_effect=fake_metric),
            ):
                calculator = cls()
                result = calculator.calculate(
                    prediction="prediction", reference="reference"
                )
                asynchronous = asyncio.run(
                    calculator.calculate_async(
                        prediction="prediction", reference="reference"
                    )
                )
                self.assertEqual(result.value, expected)
                self.assertEqual(result.model_dump(), asynchronous.model_dump())
                self.assertEqual(
                    (result.metadata or {})["metric_provenance"],
                    calculator.implementation.provenance,
                )

    def test_bertscore_cleanup_loads_the_same_pinned_implementation(self):
        with patch(f"{MODULE}.evaluate.load", side_effect=fake_metric) as load:
            evaluator = get_bertscore_evaluator()
            assert evaluator.implementation is not None
            expected = evaluator.implementation.provenance

            async def score_once():
                state = TaskState(
                    model=ModelName("mockllm/model"),
                    sample_id=1,
                    epoch=1,
                    input="source",
                    messages=[],
                    output=ModelOutput.from_content(
                        model="fixture", content="prediction"
                    ),
                )
                return await cast(Scorer, evaluator.scorer)(state, Target("reference"))

            first = asyncio.run(score_once())
            assert evaluator.cleanup_fun is not None
            with patch.dict(sys.modules, {"torch": Mock()}):
                evaluator.cleanup_fun()
            second = asyncio.run(score_once())
            assert first is not None and second is not None and second.metadata
            self.assertEqual(first.model_dump(), second.model_dump())
            self.assertEqual(second.metadata["metric_provenance"], expected)
            self.assertEqual(load.call_count, 2)
            self.assertEqual(load.call_args_list[0], load.call_args_list[1])

    def test_corpus_bleu_rejects_missing_or_mixed_implementations(self):
        implementation = MetricImplementation.resolve("bleu")
        for provenance in (
            None,
            {},
            replace(implementation, revision="0" * 40).provenance,
        ):
            with (
                self.subTest(provenance=provenance),
                patch(f"{MODULE}.evaluate.load") as load,
            ):
                score = Score(
                    value=0.75,
                    metadata={
                        "prediction": "prediction",
                        "reference": "reference",
                        "metric_provenance": provenance,
                    },
                )
                with self.assertRaisesRegex(ValueError, "same recorded"):
                    bleu_metric(implementation=implementation)(
                        [SampleScore(score=score)]
                    )
                load.assert_not_called()
        with patch(f"{MODULE}.evaluate.load", side_effect=fake_metric):
            score = BleuPrecisionScoreCalculator().calculate(
                prediction="prediction", reference="reference"
            )
            self.assertEqual(
                bleu_metric(implementation=implementation)([SampleScore(score=score)]),
                0.25,
            )

    def test_pipeline_persists_identity_and_separate_logs_without_regenerating(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp)),
            patch(f"{MODULE}.evaluate.load", side_effect=fake_metric),
        ):
            with patch(f"{MODULE}.version", return_value="first"):
                first = experiment(get_bleu_evaluator())
            with patch(f"{MODULE}.version", return_value="second"):
                second = experiment(get_bleu_evaluator())
            project = Project("provenance")
            project.generation_log_path.mkdir(parents=True)
            path = project.generation_log_path / "2026_task_original.json"
            write_eval_log(source_log(), location=str(path))
            project.update_record(
                first.generation_record,
                ResultRecord(status="success", log_location=str(path)),
            )
            Pipeline([first, second], project).evaluate(show_progress=False)
            restored = Project("provenance")
            self.assertEqual(len(restored.records.generation), 1)
            self.assertEqual(len(restored.records.evaluation), 2)
            paths = set()
            for exp in (first, second):
                saved = restored.get_record(exp.evaluation_record)
                assert saved is not None
                self.assertEqual(saved.status, "success")
                paths.add(saved.log_location)
                log = restored.get_log(exp.evaluation_record)
                assert log is not None and log.samples and log.samples[0].scores
                result = next(iter(log.samples[0].scores.values()))
                assert result.metadata
                self.assertEqual(
                    result.metadata["metric_provenance"],
                    json.loads(exp.evaluation_record.evaluator_provenance_json),
                )
            self.assertEqual(len(paths), 2)
            before = Path(path).read_bytes()
            Pipeline([first], restored).evaluate(show_progress=False)
            self.assertEqual(Path(path).read_bytes(), before)

    def test_all_analysers_reject_mixed_or_legacy_versions_before_reporting(self):
        with patch(f"{MODULE}.evaluate.load", side_effect=fake_metric):
            record = experiment(get_rouge_evaluator()).evaluation_record
        for other in ("", "different version"):
            logs = {
                record: source_log(),
                record.model_copy(
                    update={"evaluator_provenance_json": other}
                ): source_log(),
            }
            project = Mock(spec=Project)
            project.get_logs.return_value = logs
            for cls in (
                TabularResultAnalyser,
                MetaResultAnalyser,
                MetricCorrelationAnalyser,
            ):
                with (
                    self.subTest(analyser=cls.__name__, legacy=not other),
                    self.assertRaisesRegex(
                        ValueError, "different metric implementations"
                    ),
                ):
                    if cls is TabularResultAnalyser:
                        TabularResultAnalyser()(project)
                    elif cls is MetricCorrelationAnalyser:
                        MetricCorrelationAnalyser()(project)
                    else:
                        MetaResultAnalyser()(project, expected_tiers=[0, 1])
        check_metric_implementations({record: source_log()})
        legacy = record.model_copy(update={"evaluator_provenance_json": ""})
        check_metric_implementations({legacy: source_log()})

    def test_different_evaluator_names_cannot_hide_colliding_inner_metrics(self):
        with patch(f"{MODULE}.evaluate.load", side_effect=fake_metric):
            record = experiment(get_rouge_evaluator()).evaluation_record
        other = record.model_copy(
            update={"evaluator_name": "other", "evaluator_provenance_json": "different"}
        )
        logs = {}
        for key in (record, other):
            log = source_log()
            assert log.samples
            log.samples[0].scores = {key.evaluator_name: Score(value={"ROUGE-1": 1.0})}
            logs[key] = log
        with self.assertRaisesRegex(ValueError, "different metric implementations"):
            check_metric_implementations(logs)


@unittest.skipUnless(
    os.environ.get("EVALSENSE_TEST_NATIVE_METRICS") == "1",
    "requires cached metric scripts or Hub access",
)
class NativeImplementationTests(unittest.TestCase):
    def test_real_bleu_and_rouge_sources_and_values(self):
        for name, cls in (
            ("bleu", BleuPrecisionScoreCalculator),
            ("rouge", RougeScoreCalculator),
        ):
            with self.subTest(metric=name):
                calculator = cls()
                result = calculator.calculate(
                    prediction="one two three four", reference="one two three four"
                )
                self.assertTrue(
                    all(value == 1.0 for value in result.value.values())
                    if isinstance(result.value, dict)
                    else result.value == 1.0
                )
                assert result.metadata
                self.assertEqual(
                    result.metadata["metric_provenance"]["source_hash"],
                    calculator.implementation.source_hash,
                )

    @unittest.skipUnless(
        os.environ.get("EVALSENSE_TEST_BERTSCRIPT") == "1",
        "requires transformers extra",
    )
    def test_real_bertscore_script_loads_without_downloading_model(self):
        implementation = MetricImplementation.resolve("bertscore")
        module = implementation.load()
        self.assertEqual(module._hash, implementation.source_hash)
        self.assertFalse(hasattr(module, "cached_bertscorer"))


if __name__ == "__main__":
    unittest.main()
