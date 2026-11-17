import importlib.util
import unittest
from unittest.mock import patch

from inspect_ai.log import EvalMetric, EvalResults, EvalScore

from test_metric_correlation import log, record, sample


def mean_result(name: str, value: float) -> EvalResults:
    return EvalResults(
        total_samples=2,
        completed_samples=2,
        scores=[
            EvalScore(
                name=name,
                scorer=name,
                params={},
                metrics={"mean": EvalMetric(name="mean", value=value, params={})},
            )
        ],
    )


@unittest.skipUnless(importlib.util.find_spec("gradio"), "webui extra not installed")
class StandardSummaryTests(unittest.TestCase):
    def test_summary_without_samples_and_next_project_with_samples(self):
        from matplotlib import pyplot as plt
        from matplotlib.figure import Figure

        from evalsense.webui.components.results import load_project

        results = mean_result("quality", 0.75)
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

    def test_ambiguous_correlations_keep_both_summary_values(self):
        from evalsense.webui.components.results import load_project

        logs = {}
        for version, values in (("a", [0.0, 0.5]), ("b", [0.5, 1.0])):
            entry = log([sample(i, {"rating": v}) for i, v in enumerate(values)])
            entry.results = mean_result(f"summary-{version}", sum(values) / len(values))
            logs[record(evaluator_fingerprint=version)] = entry

        with (
            patch("evalsense.webui.components.results.Project") as project,
            patch("gradio.Warning") as warning,
        ):
            project.return_value.get_logs.return_value = logs
            summary, plot = load_project("ambiguous-plot", False)

        self.assertEqual(summary["summary-a/mean"].tolist(), [0.25])
        self.assertEqual(summary["summary-b/mean"].tolist(), [0.75])
        self.assertFalse(plot["visible"])
        self.assertIsNone(plot["value"])
        warning.assert_called_once()
        self.assertIn("Metric labels are ambiguous", warning.call_args.args[0])

    def test_incompatible_summary_provenance_still_fails(self):
        import gradio as gr
        from evalsense.webui.components.results import load_project

        entry = log([])
        entry.results = mean_result("quality", 0.75)
        logs = {
            record(evaluator_provenance_json='{"revision":"a"}'): entry,
            record(evaluator_provenance_json='{"revision":"b"}'): entry,
        }
        with (
            patch("evalsense.webui.components.results.Project") as project,
            patch("gradio.Warning") as warning,
        ):
            project.return_value.get_logs.return_value = logs
            with self.assertRaisesRegex(gr.Error, "different metric implementations"):
                load_project("mixed-provenance", False)
            warning.assert_not_called()
