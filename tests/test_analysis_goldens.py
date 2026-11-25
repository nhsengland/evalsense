"""Complete synthetic observations with independently known correlations."""

import math
from typing import Any
import unittest
from unittest.mock import Mock

import polars as pl
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec
from inspect_ai.scorer import Score

from evalsense.datasets import DatasetRecord
from evalsense.evaluation import EvaluationRecord, ResultRecord
from evalsense.generation import ModelRecord
from evalsense.workflow import Project
from evalsense.workflow.project import ProjectRecords
from evalsense.workflow.analysers import MetaResultAnalyser, MetricCorrelationAnalyser


def record(generator: str = "generate") -> EvaluationRecord:
    return EvaluationRecord(
        dataset_record=DatasetRecord(name="synthetic", version="1", splits=("test",)),
        model_record=ModelRecord(name="mockllm/model"),
        task_name="task",
        generator_name=generator,
        evaluator_name="fixture",
    )


def log(samples: list[EvalSample]) -> EvalLog:
    return EvalLog(
        status="success",
        eval=EvalSpec(
            created="2026-01-01T00:00:00Z",
            task="task",
            model="mockllm/model",
            dataset=EvalDataset(name="synthetic"),
            config=EvalConfig(),
            run_id="run",
            task_id="task",
            eval_id="source",
        ),
        samples=samples,
    )


def project(logs: dict[EvaluationRecord, EvalLog]) -> Project:
    fixture = Mock(spec=Project)
    fixture.get_logs.return_value = logs
    fixture.records = ProjectRecords()
    fixture.records.evaluation = {key: ResultRecord(status="success") for key in logs}
    fixture.get_log.side_effect = logs.get
    return fixture


class AnalysisGoldenTests(unittest.TestCase):
    def test_complete_rankings_reproduce_known_correlations(self):
        samples = [
            EvalSample(
                id=i,
                epoch=1,
                input="input",
                target="target",
                scores={"a": Score(value=a), "b": Score(value=b), "c": Score(value=c)},
            )
            for i, (a, b, c) in enumerate(zip([1, 2, 3], [3, 2, 1], [1, 1, 2]))
        ]
        fixture = project({record(): log(samples)})
        matrix = MetricCorrelationAnalyser()(
            fixture, return_plot=False
        ).correlation_matrix
        self.assertIsInstance(matrix, pl.DataFrame)
        self.assertAlmostEqual(matrix.filter(pl.col("Metric") == "a")["b"].item(), -1)
        self.assertAlmostEqual(
            matrix.filter(pl.col("Metric") == "a")["c"].item(), math.sqrt(3) / 2
        )
        self.assertAlmostEqual(matrix.filter(pl.col("Metric") == "a")["a"].item(), 1)

    def test_pearson_uses_values_instead_of_ranks(self):
        samples = [
            EvalSample(
                id=i,
                epoch=1,
                input="input",
                target="target",
                scores={"a": Score(value=a), "b": Score(value=b)},
            )
            for i, (a, b) in enumerate(zip([1, 2, 3], [1, 2, 4]))
        ]
        matrix = MetricCorrelationAnalyser()(
            project({record(): log(samples)}), corr_method="pearson", return_plot=False
        ).correlation_matrix
        # Centred cross-product 3; sums of squares 2 and 14/3.
        self.assertAlmostEqual(
            matrix.filter(pl.col("Metric") == "a")["b"].item(), 3 * math.sqrt(3 / 28)
        )

    def test_complete_meta_design_averages_samples_and_respects_direction(self):
        logs = {}
        for tier in range(3):
            logs[record(f"tier {tier}")] = log(
                [
                    EvalSample(
                        id=id,
                        epoch=1,
                        input="input",
                        target="target",
                        metadata={"tier": tier},
                        scores={"quality": Score(value=values[tier])},
                    )
                    for id, values in (("first", [1, 2, 3]), ("second", [3, 1, 2]))
                ]
            )
        fixture = project(logs)
        # Each sample has all planned tiers. Their correlations are +1 and -1/2.
        # Supplying the design also supports the stricter evidence-aware API.
        options: dict[str, Any] = {"expected_tiers": [0, 1, 2]}
        for lower_is_better, expected in ((False, 0.25), (True, -0.25)):
            with self.subTest(lower_is_better=lower_is_better):
                result = MetaResultAnalyser()(
                    fixture,
                    meta_tier_field="tier",
                    lower_tier_is_better=lower_is_better,
                    **options,
                )
                self.assertAlmostEqual(result["avg_correlation"].item(), expected)


if __name__ == "__main__":
    unittest.main()
