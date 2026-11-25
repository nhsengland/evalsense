"""Exercise the weighted rating contract through the public scorer and Inspect."""

from typing import cast
import unittest
from unittest.mock import patch

from inspect_ai.model import GenerateConfig, Logprob, ModelOutput, TopLogprob, get_model
from inspect_ai.scorer import Target

from test_g_eval_scoring import scorer_for, task_state
from test_g_eval_weighted_ratings import rating_options, weighted_output


def split_rating_output() -> ModelOutput:
    output = weighted_output("10", "1", [("1", 0.7), ("9", 0.3)])
    assert output.choices[0].logprobs
    output.choices[0].logprobs.content.append(
        Logprob(
            token="0",
            logprob=0.0,
            top_logprobs=[TopLogprob(token="0", logprob=0.0)],
        )
    )
    return output


class WeightedScorerTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_rating_tokens_keep_the_expected_raw_and_normalised_score(
        self,
    ):
        for prefix in ("", " ", "▁"):
            for normalise in (False, True):
                with self.subTest(prefix=prefix, normalise=normalise):
                    choices = [
                        (prefix + text, probability)
                        for text, probability in rating_options({9: 0.3, 10: 0.7})
                    ]
                    model = get_model(
                        "mockllm/judge",
                        custom_outputs=[weighted_output("9", prefix + "9", choices)],
                    )
                    result = await scorer_for(
                        model, logprobs=True, normalise=normalise
                    )(task_state(), Target("reference"))
                    assert result is not None and result.metadata
                    self.assertAlmostEqual(
                        cast(float, result.value), 8.7 / 9 if normalise else 9.7
                    )
                    self.assertAlmostEqual(result.metadata["raw_score"], 9.7)
                    self.assertEqual(result.metadata["output_text"], "9")
                    self.assertEqual(
                        result.metadata["estimator"], "weighted_rating_tokens"
                    )
                    self.assertEqual(result.answer, "candidate")

    async def test_prefix_and_split_ratings_fail_then_a_complete_distribution_can_retry(
        self,
    ):
        cases = (
            (
                weighted_output("9", "9", [("9", 0.3), ("1", 0.7)]),
                "complete single-token alternative",
            ),
            (split_rating_output(), "exactly one output token"),
        )
        for invalid, reason in cases:
            with self.subTest(response=invalid.completion):
                model = get_model(
                    "mockllm/judge",
                    custom_outputs=[
                        invalid,
                        weighted_output("9", "9", rating_options({9: 0.3, 10: 0.7})),
                    ],
                )
                run = scorer_for(model, logprobs=True, normalise=False)
                with self.assertRaisesRegex(ValueError, reason):
                    await run(task_state(), Target("reference"))
                recovered = await run(task_state(), Target("reference"))
                assert recovered is not None
                self.assertAlmostEqual(cast(float, recovered.value), 9.7)

    async def test_weighted_limits_reach_the_provider_over_inherited_model_settings(
        self,
    ):
        inherited = GenerateConfig(max_tokens=64, num_choices=3, logprobs=False)
        model = get_model(
            "mockllm/judge",
            config=inherited,
            custom_outputs=[
                weighted_output("9", "9", rating_options({9: 0.3, 10: 0.7}))
            ],
        )
        with patch.object(model.api, "generate", wraps=model.api.generate) as provider:
            result = await scorer_for(model, logprobs=True)(
                task_state(), Target("reference")
            )
        assert result is not None
        provider.assert_awaited_once()
        actual = provider.call_args.kwargs["config"]
        self.assertEqual(actual.max_tokens, 1)
        self.assertEqual(actual.num_choices, 1)
        self.assertTrue(actual.logprobs)
        self.assertEqual(actual.top_logprobs, 20)
        self.assertEqual(model.config.max_tokens, 64)
        self.assertEqual(model.config.num_choices, 3)

    async def test_explicit_unweighted_mode_accepts_a_multi_token_integer(self):
        model = get_model(
            "mockllm/judge",
            config=GenerateConfig(max_tokens=64),
            custom_outputs=[split_rating_output()],
        )
        with patch.object(model.api, "generate", wraps=model.api.generate) as provider:
            result = await scorer_for(model, logprobs=False)(
                task_state(), Target("reference")
            )
        assert result is not None and result.metadata
        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.metadata["raw_score"], 10)
        self.assertEqual(result.metadata["estimator"], "sampled_rating")
        self.assertNotIn("rating_probabilities", result.metadata)
        actual = provider.call_args.kwargs["config"]
        self.assertEqual(actual.max_tokens, 64)
        self.assertFalse(actual.logprobs)
