import importlib.util
import unittest
from unittest.mock import patch

from inspect_ai.log import EvalMetric, EvalResults, EvalScore

from test_metric_correlation import log, record, sample


@unittest.skipUnless(importlib.util.find_spec("gradio"), "webui extra not installed")
class StandardSummaryTests(unittest.TestCase):
    def test_summary_without_samples_and_next_project_with_samples(self):
        from matplotlib import pyplot as plt
        from matplotlib.figure import Figure

        from evalsense.webui.components.results import load_project

        results = EvalResults(
            total_samples=2,
            completed_samples=2,
            scores=[
                EvalScore(
                    name="quality",
                    scorer="quality",
                    params={},
                    metrics={"mean": EvalMetric(name="mean", value=0.75, params={})},
                )
            ],
        )
        aggregate_only = log([])
        aggregate_only.results = results
        with_samples = log([sample(1, {"quality": 0.5}), sample(2, {"quality": 1.0})])
        with_samples.results = results

        with patch("evalsense.webui.components.results.Project") as project:
            project.return_value.get_logs.return_value = {record(): aggregate_only}
            summary, hidden = load_project("aggregate", False)
            self.assertEqual(summary["quality/mean"].tolist(), [0.75])
            self.assertFalse(hidden["visible"])
            self.assertIsNone(hidden["value"])

            project.return_value.get_logs.return_value = {record(): with_samples}
            summary, shown = load_project("with-samples", False)
            self.assertEqual(summary["quality/mean"].tolist(), [0.75])
            self.assertTrue(shown["visible"])
            self.assertIsInstance(shown["value"], Figure)
            self.addCleanup(plt.close, shown["value"])
