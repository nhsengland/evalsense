"""Regression tests for the standard experiment summary."""

import unittest

import polars as pl
from inspect_ai.log import EvalMetric, EvalResults, EvalScore

from evalsense.datasets import DatasetRecord
from evalsense.generation import ModelRecord
from evalsense.workflow.analysers import TabularResultAnalyser
from test_meta_evidence import log, project, record


def score(value, name="quality", metric="mean", **options):
    return EvalScore(
        name=name,
        scorer=options.pop("scorer", name),
        metrics={metric: EvalMetric(name=metric, value=value)},
        **options,
    )


def summary(*scores):
    return log([], results=EvalResults(scores=list(scores)))


class TabularResultTests(unittest.TestCase):
    def test_generation_settings_keep_scores_on_distinct_rows(self):
        cold, warm = [
            record(
                0,
                model_record=ModelRecord(
                    name="model", generation_args_json=f'{{"temperature":{temp}}}'
                ),
            )
            for temp in (0, 1)
        ]
        logs = {cold: summary(score(0.2)), warm: summary(score(0.9))}
        for ordered in (logs, dict(reversed(logs.items()))):
            table = TabularResultAnalyser[pl.DataFrame]()(project(ordered))
            self.assertCountEqual(
                table.select("generation_id", "quality/mean").rows(),
                [(cold.generation_id, 0.2), (warm.generation_id, 0.9)],
            )
        self.assertNotEqual(cold.generation_id, warm.generation_id)

    def test_generation_id_covers_the_persisted_generation_fields(self):
        original = record(0)
        alternatives = [
            {
                "dataset_record": DatasetRecord(
                    name="data", version="v2", splits=("test",)
                )
            },
            {"model_record": ModelRecord(name="model", model_args_json='{"seed":17}')},
            {"experiment_name": "replication"},
            {"generator_fingerprint": "new-solver"},
            {"task_fingerprint": "new-preprocessor"},
        ]
        for changed in alternatives:
            with self.subTest(changed=changed):
                self.assertNotEqual(
                    original.generation_id, record(0, **changed).generation_id
                )
        splits_a, splits_b = [
            record(
                0,
                dataset_record=DatasetRecord(name="data", version="v1", splits=splits),
            )
            for splits in (("a, b",), ("a", "b"))
        ]
        self.assertNotEqual(splits_a.generation_id, splits_b.generation_id)

    def test_distinct_evaluators_share_the_generation_row(self):
        first, second = record(0), record(0, evaluator_name="another judge")
        table = TabularResultAnalyser[pl.DataFrame]()(
            project(
                {first: summary(score(0.2)), second: summary(score(0.9, name="other"))}
            )
        )
        self.assertEqual(first.generation_id, first.generation_record.generation_id)
        self.assertEqual(first.generation_id, second.generation_id)
        self.assertEqual(
            table.select("generation_id", "quality/mean", "other/mean").rows(),
            [(first.generation_id, 0.2, 0.9)],
        )

    def test_repeated_metric_for_one_generation_raises_even_for_equal_values(self):
        for later in (0.2, 0.9):
            with self.subTest(later=later):
                logs = {record(0): summary(score(0.2), score(later))}
                with self.assertRaisesRegex(ValueError, "Duplicate.*quality/mean"):
                    TabularResultAnalyser()(project(logs))

    def test_distinct_evaluator_settings_keep_their_own_metric_columns(self):
        logs = {
            record(0, evaluator_fingerprint=fp): summary(score(value))
            for fp, value in (("configuration-a", 0.2), ("configuration-b", 0.9))
        }
        analyser = TabularResultAnalyser[pl.DataFrame]()
        table = analyser(project(logs))
        cols = [col for col in table.columns if col.startswith("quality/mean [judge, ")]
        self.assertEqual(len(cols), 2)
        self.assertCountEqual(table.select(cols).row(0), [0.2, 0.9])
        reversed_table = analyser(project(dict(reversed(logs.items()))))
        self.assertEqual(table.select(cols).rows(), reversed_table.select(cols).rows())

    def test_slashes_in_scorer_and_metric_names_do_not_alias_cells(self):
        data = project({record(0): summary(score(1, "a/b", "c"), score(2, "a", "b/c"))})
        table = TabularResultAnalyser[pl.DataFrame]()(data)
        cols = [col for col in table.columns if col.startswith("a/b/c [")]
        self.assertCountEqual(table.select(cols).row(0), [1, 2])

    def test_different_aggregates_with_one_display_name_remain_distinct(self):
        for other in (
            score(2, scorer="another"),
            score(2, reducer="median"),
            score(2, params={"k": 2}),
        ):
            with self.subTest(other=other):
                table = TabularResultAnalyser[pl.DataFrame]()(
                    project({record(0): summary(score(1), other)})
                )
                labels = [
                    col for col in table.columns if col.startswith("quality/mean [")
                ]
                self.assertCountEqual(table.select(labels).row(0), [1, 2])

    def test_disjoint_generations_do_not_share_a_mislabelled_column(self):
        logs = {
            record(0, evaluator_name="left"): summary(score(0.2)),
            record(1, evaluator_name="right"): summary(score(0.9)),
        }
        table = TabularResultAnalyser[pl.DataFrame]()(project(logs))
        for name, value in (("left", 0.2), ("right", 0.9)):
            cols = [
                col for col in table.columns if col.startswith(f"quality/mean [{name},")
            ]
            self.assertEqual(len(cols), 1)
            self.assertCountEqual(table[cols[0]].to_list(), [value, None])

    def test_dataset_revision_and_experiment_name_are_readable(self):
        runs = [
            record(
                0,
                experiment_name=name,
                dataset_record=DatasetRecord(
                    name="data", version=version, splits=("test",)
                ),
            )
            for name, version in ((None, "v1"), ("replication", "v2"))
        ]
        logs = {run: summary(score(0.5)) for run in runs}
        table = TabularResultAnalyser[pl.DataFrame]()(project(logs))
        self.assertCountEqual(
            table.select("dataset_version", "experiment_name", "generation_id").rows(),
            [
                ("v1", None, runs[0].generation_id),
                ("v2", "replication", runs[1].generation_id),
            ],
        )

    def test_no_metrics_returns_the_identity_columns_in_both_formats(self):
        expected = [
            "dataset",
            "splits",
            "task",
            "generator",
            "model",
            "generation_id",
            "dataset_version",
            "experiment_name",
        ]
        for logs in ({}, {record(0): log([])}, {record(0): summary()}):
            data = project(logs)
            polars = TabularResultAnalyser[pl.DataFrame]()(data)
            pandas = TabularResultAnalyser(output_format="pandas")(data)
            self.assertEqual(polars.columns, expected)
            self.assertEqual(polars.height, 0)
            self.assertEqual(list(pandas.columns), expected)
            self.assertEqual(len(pandas), 0)
            self.assertTrue(all(dtype == pl.String for dtype in polars.dtypes))

    def test_missing_integer_counts_keep_their_precision_beside_float_metrics(self):
        count = 2**53 + 1
        logs = {
            record(0): summary(score(count, "events", "count"), score(0.25)),
            record(1): summary(score(0.5)),
        }
        polars = TabularResultAnalyser[pl.DataFrame]()(project(logs))
        self.assertEqual(polars["events/count"].drop_nulls().to_list(), [count])
        pandas = TabularResultAnalyser(output_format="pandas")(project(logs))
        observed = pandas["events/count"].dropna().iloc[0]
        self.assertEqual(int(observed), count)
