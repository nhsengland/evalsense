"""Weighted configuration from the UI must preserve the requested token budget."""

import importlib.util
from typing import cast
import unittest

from evalsense.evaluation.evaluators.g_eval import GEvalScorerFactory


@unittest.skipUnless(importlib.util.find_spec("gradio"), "webui extra not installed")
class GEvalUIWeightTests(unittest.TestCase):
    def test_probability_budget_rejects_rounding_and_non_positive_values(self):
        from evalsense.webui.configurators.evaluators.g_eval import GEvalConfigurator

        parser = next(
            item["parser"]
            for item in GEvalConfigurator().input_widget()
            if item["input_name"] == "top_logprobs"
        )
        assert parser is not None
        for invalid in (0, -1, 3.5, float("inf"), float("nan"), True, None, "20"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                parser(invalid)
        for valid in (1, 5.0, 20):
            self.assertIs(type(parser(valid)), int)
            self.assertEqual(parser(valid), valid)

    def test_probability_budget_reaches_the_factory_without_truncation(self):
        from evalsense.webui.configurators.evaluators.g_eval import GEvalConfigurator

        configurator = GEvalConfigurator()
        parser = next(
            item["parser"]
            for item in configurator.input_widget()
            if item["input_name"] == "top_logprobs"
        )
        assert parser is not None
        evaluator = configurator.instantiate_evaluator(
            model_name="mockllm/judge",
            prompt_template="Rate {prediction}",
            min_score=1,
            max_score=5,
            logprobs=True,
            top_logprobs=parser(5.0),
        )
        factory = cast(GEvalScorerFactory, evaluator.scorer)
        self.assertEqual(factory.top_logprobs, 5)
        self.assertTrue(factory.logprobs)

    def test_weighted_budget_must_cover_the_entire_rating_scale(self):
        from evalsense.webui.configurators.evaluators.g_eval import GEvalConfigurator

        configurator = GEvalConfigurator()
        with self.assertRaisesRegex(ValueError, "top_logprobs"):
            configurator.instantiate_evaluator(
                model_name="mockllm/judge",
                prompt_template="Rate {prediction}",
                min_score=1,
                max_score=5,
                logprobs=True,
                top_logprobs=4,
            )
        sampled = configurator.instantiate_evaluator(
            model_name="mockllm/judge",
            prompt_template="Rate {prediction}",
            min_score=1,
            max_score=10,
            logprobs=False,
            top_logprobs=4,
        )
        self.assertFalse(cast(GEvalScorerFactory, sampled.scorer).logprobs)
