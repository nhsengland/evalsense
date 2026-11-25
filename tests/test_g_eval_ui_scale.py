"""The web UI must not silently truncate the declared rating scale."""

import importlib.util
from typing import cast
import unittest

from evalsense.evaluation.evaluators.g_eval import GEvalScorerFactory


@unittest.skipUnless(importlib.util.find_spec("gradio"), "webui extra not installed")
class GEvalUIScaleTests(unittest.TestCase):
    def test_numeric_widgets_reject_fractional_and_non_finite_bounds(self):
        from evalsense.webui.configurators.evaluators.g_eval import GEvalConfigurator

        widgets = GEvalConfigurator().input_widget()
        for key in ("min_score", "max_score"):
            parser = next(
                item["parser"] for item in widgets if item["input_name"] == key
            )
            assert parser is not None
            for invalid in (3.5, -1, float("inf"), float("nan"), True, None, "5"):
                with (
                    self.subTest(bound=key, value=invalid),
                    self.assertRaises(ValueError),
                ):
                    parser(invalid)
            for valid in (0, 1, 5.0, 10):
                self.assertIs(type(parser(valid)), int)
                self.assertEqual(parser(valid), valid)

    def test_widget_values_reach_the_factory_without_changing_the_scale(self):
        from evalsense.webui.configurators.evaluators.g_eval import GEvalConfigurator

        configurator = GEvalConfigurator()
        parsers = {
            item["input_name"]: item["parser"] for item in configurator.input_widget()
        }
        low, high = parsers["min_score"], parsers["max_score"]
        assert low is not None and high is not None
        evaluator = configurator.instantiate_evaluator(
            model_name="mockllm/judge",
            prompt_template="Rate {prediction}",
            min_score=low(0.0),
            max_score=high(5.0),
            logprobs=False,
        )
        factory = cast(GEvalScorerFactory, evaluator.scorer)
        self.assertEqual((factory.min_score, factory.max_score), (0, 5))
        with self.assertRaises(ValueError):
            configurator.instantiate_evaluator(
                model_name="mockllm/judge",
                prompt_template="Rate {prediction}",
                min_score=low(5.0),
                max_score=high(1.0),
            )
