import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalSample,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.scorer import Score

from evalsense.datasets import DatasetRecord
from evalsense.evaluation import EvaluationRecord, ResultRecord
from evalsense.generation import ModelRecord
from evalsense.workflow import Project
from evalsense.workflow.analysers.metric_correlation_analyser import (
    CorrelationResults,
    MetricCorrelationAnalyser,
)


def record(evaluator="judge", **kwargs) -> EvaluationRecord:
    values: dict[str, Any] = dict(
        dataset_record=DatasetRecord(name="data", version="v1", splits=("test",)),
        model_record=ModelRecord(name="model"),
        task_name="task",
        generator_name="generate",
        evaluator_name=evaluator,
    )
    values.update(kwargs)
    return EvaluationRecord(**values)


def sample(id, values, epoch=1) -> EvalSample:
    return EvalSample(
        id=id,
        epoch=epoch,
        input="input",
        target="target",
        scores={name: Score(value=value) for name, value in values.items()},
    )


def log(samples, source="source") -> EvalLog:
    return EvalLog(
        status="success",
        eval=EvalSpec(
            created="2026-01-01T00:00:00Z",
            task="task",
            model="model",
            dataset=EvalDataset(name="data"),
            config=EvalConfig(),
            eval_id=source,
            run_id="fixture-run",
            task_id=source,
        ),
        samples=samples,
    )


def analyse(logs, **kwargs) -> CorrelationResults[pl.DataFrame]:
    project = Mock(spec=Project)
    project.get_logs.return_value = logs
    analyser = MetricCorrelationAnalyser[CorrelationResults[pl.DataFrame]]()
    return analyser(project, return_plot=False, **kwargs)


def cell(table, left, right):
    return table.filter(pl.col("Metric") == left).get_column(right).item()


class CorrelationTests(unittest.TestCase):
    def test_equal_lengths_do_not_imply_shared_samples(self):
        result = analyse(
            {
                record(): log(
                    [
                        sample(1, {"a": 10}),
                        sample(2, {"a": 20}),
                        sample(3, {"b": 10}),
                        sample(4, {"b": 20}),
                    ]
                )
            }
        )
        self.assertTrue(math.isnan(cell(result.correlation_matrix, "a", "b")))
        self.assertEqual(cell(result.paired_counts, "a", "b"), 0)

    def test_partial_overlap_uses_only_matched_pairs(self):
        result = analyse(
            {
                record(): log(
                    [
                        sample(1, {"a": -100}),
                        sample(2, {"a": 1, "b": 2}),
                        sample(3, {"a": 2, "b": 1}),
                        sample(4, {"b": 100}),
                    ]
                )
            }
        )
        self.assertAlmostEqual(cell(result.correlation_matrix, "a", "b"), -1)
        self.assertEqual(cell(result.paired_counts, "a", "b"), 2)
        self.assertEqual(cell(result.paired_counts, "a", "a"), 3)

    def test_separate_evaluators_join_the_same_source(self):
        result = analyse(
            {
                record("left"): log([sample(1, {"a": 1}), sample(2, {"a": 2})]),
                record("right"): log([sample(2, {"b": 1}), sample(1, {"b": 2})]),
            }
        )
        self.assertAlmostEqual(cell(result.correlation_matrix, "a", "b"), -1)
        self.assertEqual(cell(result.paired_counts, "a", "b"), 2)

    def test_different_attempts_do_not_join(self):
        result = analyse(
            {
                record("left"): log([sample(1, {"a": 1}), sample(2, {"a": 2})], "old"),
                record("right"): log([sample(1, {"b": 1}), sample(2, {"b": 2})], "new"),
            }
        )
        self.assertEqual(cell(result.paired_counts, "a", "b"), 0)

    def test_full_generation_identity_is_preserved(self):
        differences = [
            dict(model_record=ModelRecord(name="another model")),
            dict(
                model_record=ModelRecord(
                    name="model", generation_args_json='{"temperature":1}'
                )
            ),
            dict(
                dataset_record=DatasetRecord(
                    name="other", version="v1", splits=("test",)
                )
            ),
            dict(
                dataset_record=DatasetRecord(
                    name="data", version="v2", splits=("test",)
                )
            ),
            dict(
                dataset_record=DatasetRecord(
                    name="data", version="v1", splits=("train",)
                )
            ),
            dict(task_name="other task"),
            dict(generator_name="other generator"),
            dict(experiment_name="other experiment"),
        ]
        for difference in differences:
            with self.subTest(difference=difference):
                result = analyse(
                    {
                        record("left"): log([sample(1, {"a": 1}), sample(2, {"a": 2})]),
                        record("right", **difference): log(
                            [sample(1, {"b": 1}), sample(2, {"b": 2})]
                        ),
                    }
                )
                self.assertEqual(cell(result.paired_counts, "a", "b"), 0)

    def test_epochs_are_paired_explicitly(self):
        result = analyse(
            {
                record("left"): log([sample(1, {"a": 1}, 1), sample(1, {"a": 2}, 2)]),
                record("right"): log([sample(1, {"b": 1}, 2), sample(1, {"b": 2}, 1)]),
            }
        )
        self.assertAlmostEqual(cell(result.correlation_matrix, "a", "b"), -1)
        self.assertEqual(cell(result.paired_counts, "a", "b"), 2)

    def test_id_types_do_not_collapse(self):
        result = analyse(
            {
                record(): log(
                    [
                        sample(1, {"a": 1, "b": 2}),
                        sample("1", {"a": 2, "b": 1}),
                    ]
                )
            }
        )
        self.assertAlmostEqual(cell(result.correlation_matrix, "a", "b"), -1)

    def test_duplicate_observation_is_rejected(self):
        for second in (1, 2):
            with (
                self.subTest(second=second),
                self.assertRaisesRegex(ValueError, "Duplicate observation"),
            ):
                analyse(
                    {record(): log([sample(1, {"a": 1}), sample(1, {"a": second})])}
                )

    def test_inner_names_are_only_qualified_when_ambiguous(self):
        result = analyse(
            {
                record(): log(
                    [
                        sample(
                            1,
                            {
                                "left": {"shared": 1, "unique": 1},
                                "right": {"shared": 2},
                            },
                        ),
                        sample(
                            2,
                            {
                                "left": {"shared": 2, "unique": 2},
                                "right": {"shared": 1},
                            },
                        ),
                    ]
                )
            }
        )
        names = result.correlation_matrix.get_column("Metric").to_list()
        self.assertIn("unique", names)
        self.assertIn("judge / left / shared", names)
        self.assertIn("judge / right / shared", names)
        self.assertAlmostEqual(
            cell(
                result.correlation_matrix,
                "judge / left / shared",
                "judge / right / shared",
            ),
            -1,
        )

    def test_evaluator_identity_is_preserved_with_same_scorer(self):
        result = analyse(
            {
                record("judge one"): log(
                    [sample(1, {"quality": 1}), sample(2, {"quality": 2})]
                ),
                record("judge two"): log(
                    [sample(1, {"quality": 2}), sample(2, {"quality": 1})]
                ),
            }
        )
        self.assertAlmostEqual(
            cell(
                result.correlation_matrix, "judge one / quality", "judge two / quality"
            ),
            -1,
        )

    def test_aliases_do_not_merge_metrics(self):
        data = {
            record(): log([sample(1, {"a": 1, "b": 2}), sample(2, {"a": 2, "b": 1})])
        }
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            analyse(data, metric_labels={"a": "same", "b": "same"})
        with self.assertRaisesRegex(ValueError, "reserved"):
            analyse(data, metric_labels={"a": "Metric"})
        result = analyse(
            data, metric_labels={"a": "A"}, method_filter_fun=lambda name: name == "a"
        )
        self.assertEqual(result.correlation_matrix.columns, ["Metric", "A"])

    def test_non_finite_pairs_do_not_create_agreement(self):
        data = [sample(1, {"a": 1, "b": 2}), sample(2, {"a": 2, "b": 1})]
        for id, value in enumerate([math.nan, math.inf, -math.inf], 3):
            data.append(sample(id, {"a": value, "b": value}))
        data.append(sample(6, {"a": 3, "b": math.nan}))
        result = analyse({record(): log(data)})
        self.assertAlmostEqual(cell(result.correlation_matrix, "a", "b"), -1)
        self.assertEqual(cell(result.paired_counts, "a", "b"), 2)
        self.assertEqual(cell(result.excluded_counts, "a", "b"), 4)

    def test_degenerate_pairs_are_undefined(self):
        for data in (
            [sample(1, {"a": 1, "b": 2})],
            [sample(1, {"a": 1, "b": 2}), sample(2, {"a": 1, "b": 3})],
        ):
            with self.subTest(data=data):
                result = analyse({record(): log(data)})
                self.assertTrue(math.isnan(cell(result.correlation_matrix, "a", "b")))
                self.assertTrue(math.isnan(cell(result.correlation_matrix, "a", "a")))

    def test_log_and_sample_permutations_are_invariant(self):
        left = [sample(1, {"a": 1}), sample(2, {"a": 3}), sample(3, {"a": 2})]
        right = [sample(3, {"b": 4}), sample(1, {"b": 1}), sample(2, {"b": 2})]
        pairs = [(record("left"), log(left)), (record("right"), log(right))]
        baseline = analyse(dict(pairs), corr_method="pearson")
        changed = analyse(
            dict(
                [
                    (pairs[1][0], log(list(reversed(right)))),
                    (pairs[0][0], log(list(reversed(left)))),
                ]
            ),
            corr_method="pearson",
        )
        self.assertTrue(baseline.correlation_matrix.equals(changed.correlation_matrix))
        assert baseline.paired_counts is not None and changed.paired_counts is not None
        self.assertTrue(baseline.paired_counts.equals(changed.paired_counts))

    def test_ties_match_known_spearman_value(self):
        result = analyse(
            {
                record(): log(
                    [
                        sample(i, {"a": a, "b": b})
                        for i, (a, b) in enumerate(zip([1, 1, 2], [1, 2, 3]))
                    ]
                )
            }
        )
        self.assertAlmostEqual(
            cell(result.correlation_matrix, "a", "b"), math.sqrt(3) / 2
        )

    def test_names_with_separators_do_not_collide(self):
        result = analyse(
            {
                record(): log(
                    [
                        sample(1, {"a__b": 1, "c": 2, "a": 1, "b__c": 2}),
                        sample(2, {"a__b": 2, "c": 1, "a": 2, "b__c": 1}),
                    ]
                )
            }
        )
        self.assertEqual(result.correlation_matrix.shape, (4, 5))

    def test_empty_and_filtered_selections(self):
        for data in ({}, {record(): log([])}, {record(): log([sample(1, {"a": "C"})])}):
            result = analyse(data)
            self.assertEqual(result.correlation_matrix.shape, (0, 1))
            self.assertIsNone(result.figure)

    def test_missing_source_id_is_rejected(self):
        data = log([sample(1, {"a": 1})])
        if hasattr(data.eval, "eval_id"):
            data.eval.eval_id = ""
        data.eval.run_id = ""
        data.eval.task_id = ""
        with self.assertRaisesRegex(ValueError, "source eval ID"):
            analyse({record(): data})

    def test_legacy_source_identity_uses_both_run_and_task(self):
        left = Mock(spec=EvalLog, samples=[sample(1, {"a": 1}), sample(2, {"a": 2})])
        right = Mock(spec=EvalLog, samples=[sample(1, {"b": 2}), sample(2, {"b": 1})])
        left.eval = SimpleNamespace(run_id="run", task_id="task")
        right.eval = SimpleNamespace(run_id="run", task_id="task")
        result = analyse({record("left"): left, record("right"): right})
        self.assertAlmostEqual(cell(result.correlation_matrix, "a", "b"), -1)
        for identity in [("other-run", "task"), ("run", "other-task")]:
            with self.subTest(identity=identity):
                right.eval = SimpleNamespace(run_id=identity[0], task_id=identity[1])
                result = analyse({record("left"): left, record("right"): right})
                self.assertEqual(cell(result.paired_counts, "a", "b"), 0)

    def test_legacy_source_identity_requires_both_ids(self):
        for run, task in [("", "task"), ("run", ""), ("", "")]:
            with self.subTest(run=run, task=task):
                data = Mock(spec=EvalLog, samples=[sample(1, {"a": 1})])
                data.eval = SimpleNamespace(run_id=run, task_id=task)
                with self.assertRaisesRegex(ValueError, "legacy run/task"):
                    analyse({record(): data})

    def test_outputs_and_pearson_plot(self):
        project = Mock(spec=Project)
        project.get_logs.return_value = {
            record(): log([sample(1, {"a": 1, "b": 2}), sample(2, {"a": 2, "b": 1})])
        }
        for fmt in ("polars", "pandas", "numpy"):
            with self.subTest(format=fmt):
                result = MetricCorrelationAnalyser(output_format=fmt)(
                    project, corr_method="pearson", return_plot=True
                )
                self.assertEqual(result.correlation_matrix.shape, (2, 3))
                self.assertEqual(result.paired_counts.shape, (2, 3))
                self.assertEqual(result.excluded_counts.shape, (2, 3))
                self.assertIn("Pearson", result.figure.axes[0].get_title())
                plt.close(result.figure)

    def test_persisted_project_pairs_copied_logs(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp)),
        ):
            project = Project("correlation")
            project.evaluation_log_path.mkdir(parents=True)
            for name, scores in [
                ("left", [(1, 1), (2, 2)]),
                ("right", [(2, 1), (1, 2)]),
            ]:
                data = log([sample(id, {name: value}) for id, value in scores])
                path = project.evaluation_log_path / f"2026_task_{name}.eval"
                write_eval_log(data, location=str(path))
                project.update_record(
                    record(name), ResultRecord(status="success", log_location=str(path))
                )
            reopened = Project("correlation")
            result = MetricCorrelationAnalyser[CorrelationResults[pl.DataFrame]]()(
                reopened, return_plot=False
            )
            self.assertAlmostEqual(cell(result.correlation_matrix, "left", "right"), -1)
            self.assertEqual(cell(result.paired_counts, "left", "right"), 2)


if __name__ == "__main__":
    unittest.main()
