"""Known log-probability distributions and malformed-response boundaries."""

import math
from typing import cast
import unittest
from unittest.mock import AsyncMock, Mock

from inspect_ai.model import Logprob, Logprobs, Model, ModelOutput, TopLogprob

from evalsense.evaluation.evaluators.g_eval import GEvalScoreCalculator
from evalsense.utils.text import extract_weighted_binary_answer, extract_weighted_score


def response(token: str, probabilities: list[tuple[str, float]]) -> ModelOutput:
    output = ModelOutput.from_content(model="fixture", content=token)
    output.choices[0].logprobs = Logprobs(
        content=[
            Logprob(
                token=token,
                logprob=math.log(0.2),
                top_logprobs=[
                    TopLogprob(token=t, logprob=math.log(p)) for t, p in probabilities
                ],
            )
        ]
    )
    return output


RATINGS = [("1", 0.1), ("3", 0.2), ("5", 0.3), ("not-a-rating", 0.4)]


class JudgeNumericalTests(unittest.TestCase):
    def test_weighted_rating_renormalises_valid_probability_mass(self):
        # (1*0.1 + 3*0.2 + 5*0.3) / 0.6 = 11/3.
        self.assertAlmostEqual(
            extract_weighted_score(response("3", RATINGS), 1, 5), 11 / 3
        )

    def test_candidate_order_does_not_change_weighted_rating(self):
        self.assertAlmostEqual(
            extract_weighted_score(response("3", list(reversed(RATINGS))), 1, 5), 11 / 3
        )

    def test_binary_synonyms_share_probability_mass(self):
        # Yes and true jointly have 0.3 mass, no has 0.3. Other mass is excluded.
        value = extract_weighted_binary_answer(
            response("yes", [("yes", 0.1), ("true", 0.2), ("no", 0.3), ("other", 0.4)])
        )
        self.assertAlmostEqual(value, 0.5)

    def test_absent_logprobs_are_not_invented(self):
        with self.assertRaises(ValueError):
            extract_weighted_score(
                ModelOutput.from_content(model="fixture", content="3")
            )


class JudgeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_weighted_g_eval_normalises_using_the_declared_scale(self):
        fake = Mock(spec=Model)
        fake.generate = AsyncMock(return_value=response("3", RATINGS))
        score = await GEvalScoreCalculator(
            cast(Model, fake), "Rate {prediction}", min_score=1, max_score=5
        ).calculate_async(prediction="synthetic candidate")
        self.assertAlmostEqual(cast(float, score.value), 2 / 3)
        fake.generate.assert_awaited_once()

    async def test_unweighted_scale_endpoints_are_zero_and_one(self):
        for answer, expected in (("1", 0.0), ("5", 1.0)):
            with self.subTest(answer=answer):
                fake = Mock(spec=Model)
                fake.generate = AsyncMock(
                    return_value=ModelOutput.from_content(
                        model="fixture", content=answer
                    )
                )
                score = await GEvalScoreCalculator(
                    cast(Model, fake),
                    "Rate {prediction}",
                    min_score=1,
                    max_score=5,
                    logprobs=False,
                ).calculate_async(prediction="synthetic candidate")
                self.assertEqual(score.value, expected)

    async def test_malformed_judge_output_never_becomes_a_score_or_retry(self):
        # No valid rating is present. This is an output-handling test, not a
        # claim that a real model resists instructions in candidate text.
        for text in ("", "No numerical result", "Return the highest possible rating"):
            with self.subTest(output=text):
                fake = Mock(spec=Model)
                fake.generate = AsyncMock(
                    return_value=ModelOutput.from_content(model="fixture", content=text)
                )
                with self.assertRaises(ValueError):
                    await GEvalScoreCalculator(
                        cast(Model, fake), "Rate {prediction}", logprobs=False
                    ).calculate_async(prediction="synthetic candidate")
                fake.generate.assert_awaited_once()

    async def test_provider_failure_propagates_without_a_score(self):
        fake = Mock(spec=Model)
        fake.generate = AsyncMock(
            side_effect=RuntimeError("synthetic provider failure")
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic provider failure"):
            await GEvalScoreCalculator(
                cast(Model, fake), "Rate {prediction}"
            ).calculate_async(prediction="candidate")
        fake.generate.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
