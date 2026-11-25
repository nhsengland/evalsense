"""Check strict integer parsing through G-Eval's public scorer factory."""

from typing import cast
import unittest
from unittest.mock import AsyncMock, Mock, patch

from inspect_ai.model import ChatMessageUser, Model, ModelName, ModelOutput, get_model
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from evalsense.evaluation.evaluators.g_eval import (
    GEvalScorerFactory,
    get_g_eval_evaluator,
)
from evalsense.generation import ModelConfig
from test_g_eval_weighted_ratings import weighted_output


def task_state() -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=1,
        epoch=1,
        input="input",
        messages=[ChatMessageUser(content="input")],
        output=ModelOutput.from_content("fixture", "candidate"),
    )


def scorer_for(model: Model, *, logprobs: bool = False, normalise: bool = True):
    evaluator = get_g_eval_evaluator(
        prompt_template="Rate {prediction} against {reference}.",
        model_config=ModelConfig("mockllm/judge"),
        logprobs=logprobs,
        normalise=normalise,
    )
    return cast(GEvalScorerFactory, evaluator.scorer).create_scorer(model)


class GEvalScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_responses_fail_before_weighting_or_normalisation(self):
        for text in ("0/10", "-3", "8.5", "Rating scale 1 to 10. Final rating 9"):
            for use_weights in (False, True):
                model = Mock(spec=Model)
                model.generate = AsyncMock(
                    side_effect=[
                        ModelOutput.from_content("fixture", text),
                        ModelOutput.from_content("fixture", "10"),
                    ]
                )
                with (
                    self.subTest(text=text, logprobs=use_weights),
                    patch(
                        "evalsense.evaluation.evaluators.g_eval.extract_weighted_score",
                        return_value=10.0,
                    ) as weighting,
                ):
                    run = scorer_for(cast(Model, model), logprobs=use_weights)
                    with self.assertRaises(ValueError):
                        await run(task_state(), Target("reference"))
                    weighting.assert_not_called()
                    self.assertEqual(model.generate.await_count, 1)

    async def test_valid_integer_ratings_keep_raw_scores_and_the_original_response(
        self,
    ):
        for text, raw in (("1", 1), (" \t8\n", 8), ("10", 10)):
            for normalise in (False, True):
                model = Mock(spec=Model)
                model.generate = AsyncMock(
                    return_value=ModelOutput.from_content("fixture", text)
                )
                result = await scorer_for(cast(Model, model), normalise=normalise)(
                    task_state(), Target("reference")
                )
                assert result is not None
                self.assertAlmostEqual(
                    cast(float, result.value), (raw - 1) / 9 if normalise else raw
                )
                self.assertEqual(result.answer, "candidate")
                assert result is not None
                assert result.metadata
                self.assertEqual(result.metadata["raw_score"], raw)
                self.assertEqual(result.metadata["output_text"], text)

    async def test_valid_weighted_ratings_retain_fractional_expectations(self):
        model = Mock(spec=Model)
        model.generate = AsyncMock(
            return_value=weighted_output("8", "8", [("8", 0.25), ("10", 0.75)])
        )
        result = await scorer_for(cast(Model, model), logprobs=True)(
            task_state(), Target("reference")
        )
        assert result is not None
        assert result.metadata
        assert result is not None
        self.assertAlmostEqual(result.metadata["raw_score"], 9.5)
        assert result is not None
        self.assertAlmostEqual(cast(float, result.value), 8.5 / 9)

    async def test_real_inspect_provider_rejects_a_fraction_then_accepts_a_new_valid_call(
        self,
    ):
        model = get_model(
            "mockllm/judge",
            custom_outputs=[
                ModelOutput.from_content("fixture", "0/10"),
                ModelOutput.from_content("fixture", "8"),
            ],
        )
        run = scorer_for(model)
        with self.assertRaises(ValueError):
            await run(task_state(), Target("reference"))
        result = await run(task_state(), Target("reference"))
        assert result is not None
        self.assertAlmostEqual(cast(float, result.value), 7 / 9)
