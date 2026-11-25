"""Reject unusable weighting requests before any judge model call."""

from typing import Any, cast
import unittest
from unittest.mock import AsyncMock, Mock

from inspect_ai.model import Model

from evalsense.evaluation.evaluators.g_eval import (
    GEvalScoreCalculator,
    GEvalScorerFactory,
    get_g_eval_evaluator,
)
from evalsense.generation import ModelConfig


class WeightingSettingsTests(unittest.TestCase):
    def test_invalid_requests_are_rejected_by_public_constructors(self):
        invalid: list[tuple[Any, Any]] = [
            (True, 0),
            (True, -1),
            (True, True),
            (True, 10.5),
            (True, float("nan")),
            (True, float("inf")),
            (True, "20"),
            (True, 9),
            ("false", 20),
            (1, 20),
            (None, 20),
        ]
        for logprobs, top_logprobs in invalid:
            options = {"logprobs": logprobs, "top_logprobs": top_logprobs}
            constructors = (
                lambda: GEvalScoreCalculator(
                    cast(Model, Mock(spec=Model)), "Rate {prediction}", **options
                ),
                lambda: GEvalScorerFactory("rating", "Rate {prediction}", **options),
                lambda: get_g_eval_evaluator(
                    prompt_template="Rate {prediction}",
                    model_config=ModelConfig("mockllm/judge"),
                    **options,
                ),
            )
            for index, build in enumerate(constructors):
                with (
                    self.subTest(options=options, constructor=index),
                    self.assertRaises(ValueError),
                ):
                    build()

    def test_weighting_budget_must_cover_the_scale_and_can_exceed_it(self):
        for budget in (10, 20, 40):
            with self.subTest(budget=budget):
                GEvalScorerFactory("rating", "Rate {prediction}", top_logprobs=budget)
        GEvalScorerFactory(
            "rating", "Rate {prediction}", min_score=8, max_score=9, top_logprobs=2
        )

    def test_unused_weighting_budget_does_not_restrict_text_only_scoring(self):
        GEvalScorerFactory(
            "rating",
            "Rate {prediction}",
            logprobs=False,
            top_logprobs=0,
            max_score=1000,
        )


class MutableWeightingSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_mutated_settings_are_checked_before_requesting_the_judge(self):
        changes: list[tuple[str, Any]] = [
            ("top_logprobs", 1),
            ("logprobs", "false"),
            ("max_score", 21),
        ]
        for attribute, value in changes:
            model = Mock(spec=Model)
            model.generate = AsyncMock()
            calculator = GEvalScoreCalculator(cast(Model, model), "Rate {prediction}")
            setattr(calculator, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                await calculator.calculate_async(prediction="candidate")
            model.generate.assert_not_called()
