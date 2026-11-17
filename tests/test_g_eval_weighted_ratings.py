"""Use the integer grammar for weighted rating tokens as well as completions."""

import math
import unittest

from inspect_ai.model import Logprob, Logprobs, ModelOutput, TopLogprob

from evalsense.utils.text import extract_weighted_binary_answer, extract_weighted_score


def weighted_output(
    completion: str, token: str, choices: list[tuple[str, float]]
) -> ModelOutput:
    output = ModelOutput.from_content("fixture", completion)
    output.choices[0].logprobs = Logprobs(
        content=[
            Logprob(
                token=token,
                logprob=math.log(0.5),
                top_logprobs=[
                    TopLogprob(token=text, logprob=math.log(probability))
                    for text, probability in choices
                ],
            )
        ]
    )
    return output


class WeightedRatingTests(unittest.TestCase):
    def test_valid_single_token_weighting_is_unchanged(self):
        for token in ("8", " 8", "▁8", "008"):
            output = weighted_output("8", token, [(token, 0.25), ("10", 0.75)])
            with self.subTest(token=token):
                self.assertAlmostEqual(extract_weighted_score(output), 9.5)

    def test_invalid_alternatives_are_not_rewritten_into_valid_ratings(self):
        for token in ("1_0", "_10", "-3", "+3", "8.5", "0/10", "٨", "1▁0"):
            output = weighted_output("8", "8", [("8", 0.25), (token, 0.75)])
            with self.subTest(token=token):
                self.assertEqual(extract_weighted_score(output), 8)

    def test_weighting_cannot_rescue_an_invalid_completion(self):
        for completion in ("0/10", "-3", "8.5", "Score: 8"):
            output = weighted_output(completion, "8", [("8", 1.0)])
            with self.subTest(completion=completion), self.assertRaises(ValueError):
                extract_weighted_score(output)

    def test_no_valid_alternatives_raises_a_validation_error(self):
        output = weighted_output("8", "8", [("0/10", 0.5), ("1_0", 0.5)])
        with self.assertRaisesRegex(ValueError, "positive finite mass"):
            extract_weighted_score(output)

    def test_malformed_sampled_token_is_not_repaired(self):
        output = weighted_output("10", "1_0", [("10", 1.0)])
        with self.assertRaisesRegex(ValueError, "identify correct token"):
            extract_weighted_score(output)

    def test_valid_binary_weighting_retains_its_existing_arithmetic(self):
        output = weighted_output("true", "true", [("true", 0.6), ("false", 0.4)])
        self.assertAlmostEqual(extract_weighted_binary_answer(output), 0.6)
