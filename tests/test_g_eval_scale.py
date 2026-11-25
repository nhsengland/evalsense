"""Reject invalid rating scales before a judge request or normalisation."""

from typing import Any, cast
import unittest
from unittest.mock import AsyncMock, Mock

from inspect_ai.model import Model, ModelOutput

from evalsense.evaluation.evaluators.g_eval import (
    GEvalScoreCalculator,
    GEvalScorerFactory,
    get_g_eval_evaluator,
)
from evalsense.generation import ModelConfig
from evalsense.utils.text import extract_score


class RatingScaleTests(unittest.TestCase):
    def test_invalid_bounds_are_rejected_by_the_parser_and_public_constructors(self):
        invalid: list[tuple[Any, Any]] = [
            (True, 10),
            (1, False),
            (1.0, 10),
            (1, 10.5),
            ("1", 10),
            (-1, 10),
            (10, 1),
            (1, float("inf")),
            (float("nan"), 10),
        ]
        for low, high in invalid:
            constructors = (
                lambda: extract_score("5", low, high),
                lambda: GEvalScoreCalculator(
                    cast(Model, Mock(spec=Model)),
                    "Rate {prediction}",
                    min_score=low,
                    max_score=high,
                ),
                lambda: GEvalScorerFactory(
                    "rating",
                    "Rate {prediction}",
                    min_score=low,
                    max_score=high,
                ),
                lambda: get_g_eval_evaluator(
                    prompt_template="Rate {prediction}",
                    model_config=ModelConfig("mockllm/judge"),
                    min_score=low,
                    max_score=high,
                ),
            )
            for index, build in enumerate(constructors):
                with (
                    self.subTest(low=low, high=high, constructor=index),
                    self.assertRaises(ValueError),
                ):
                    build()

    def test_a_single_rating_is_allowed_only_without_normalisation(self):
        self.assertEqual(extract_score("5", 5, 5), 5)
        for normalise in (True, False):
            if normalise:
                with self.assertRaisesRegex(ValueError, "min_score < max_score"):
                    GEvalScorerFactory(
                        "rating",
                        "Rate {prediction}",
                        min_score=5,
                        max_score=5,
                        normalise=normalise,
                    )
            else:
                GEvalScorerFactory(
                    "rating",
                    "Rate {prediction}",
                    min_score=5,
                    max_score=5,
                    normalise=normalise,
                )


class MutableScaleTests(unittest.IsolatedAsyncioTestCase):
    async def test_mutated_scale_is_rechecked_before_requesting_the_judge(self):
        model = Mock(spec=Model)
        model.generate = AsyncMock()
        calculator = GEvalScoreCalculator(cast(Model, model), "Rate {prediction}")
        calculator.min_score = calculator.max_score
        with self.assertRaisesRegex(ValueError, "min_score < max_score"):
            await calculator.calculate_async(prediction="candidate")
        model.generate.assert_not_called()

    async def test_unnormalised_single_rating_does_not_divide_by_zero(self):
        model = Mock(spec=Model)
        model.generate = AsyncMock(
            return_value=ModelOutput.from_content("fixture", "5")
        )
        result = await GEvalScoreCalculator(
            cast(Model, model),
            "Rate {prediction}",
            min_score=5,
            max_score=5,
            normalise=False,
            logprobs=False,
        ).calculate_async(prediction="candidate")
        self.assertEqual(result.value, 5)
