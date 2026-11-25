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
