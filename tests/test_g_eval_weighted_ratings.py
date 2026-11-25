"""Weighted ratings require complete token evidence for the configured scale."""

import math
import unittest

from inspect_ai.model import Logprob, Logprobs, ModelOutput, TopLogprob

from evalsense.utils.text import extract_weighted_binary_answer, extract_weighted_score


def weighted_output(
    completion: str, token: str, choices: list[tuple[str, float]]
) -> ModelOutput:
    output = ModelOutput.from_content("fixture", completion)
    probabilities = dict(choices)

    def log_probability(probability: float) -> float:
        return math.log(probability) if probability else -math.inf

    output.choices[0].logprobs = Logprobs(
        content=[
            Logprob(
                token=token,
                logprob=log_probability(probabilities.get(token, 0.5)),
                top_logprobs=[
                    TopLogprob(token=text, logprob=log_probability(probability))
                    for text, probability in choices
                ],
            )
        ]
    )
    return output


def rating_options(
    probabilities: dict[int, float], low: int = 1, high: int = 10
) -> list[tuple[str, float]]:
    return [
        (str(rating), probabilities.get(rating, 0.0)) for rating in range(low, high + 1)
    ]


class WeightedRatingTests(unittest.TestCase):
    def test_complete_single_token_ratings_produce_the_expected_mean(self):
        output = weighted_output("9", "9", rating_options({9: 0.3, 10: 0.7}))
        self.assertAlmostEqual(extract_weighted_score(output), 9.7)

    def test_token_prefixes_cannot_stand_in_for_the_complete_rating(self):
        output = weighted_output("9", "9", [("9", 0.3), ("1", 0.7)])
        with self.assertRaisesRegex(ValueError, "complete single-token alternative"):
            extract_weighted_score(output)

    def test_a_sampled_multi_token_rating_is_rejected(self):
        output = weighted_output("10", "1", [("1", 0.7), ("9", 0.3)])
        assert output.choices[0].logprobs
        output.choices[0].logprobs.content.append(
            Logprob(
                token="0",
                logprob=0.0,
                top_logprobs=[TopLogprob(token="0", logprob=0.0)],
            )
        )
        with self.assertRaisesRegex(ValueError, "exactly one output token"):
            extract_weighted_score(output)

    def test_supported_token_decorations_preserve_the_conditional_mean(self):
        for token in ("8", " 8", "▁8"):
            options = [
                (token if text == "8" else text, mass)
                for text, mass in rating_options({8: 0.25, 10: 0.75})
            ]
            with self.subTest(token=token):
                self.assertAlmostEqual(
                    extract_weighted_score(weighted_output("8", token, options)), 9.5
                )

    def test_invalid_alternatives_are_not_rewritten_into_valid_ratings(self):
        for token in ("1_0", "_10", "-3", "+3", "8.5", "0/10", "٨", "1▁0", "008"):
            output = weighted_output("8", "8", [("8", 0.25), (token, 0.75)])
            with self.subTest(token=token):
                self.assertEqual(extract_weighted_score(output, 8, 8), 8)

    def test_weighting_cannot_rescue_an_invalid_completion(self):
        for completion in ("0/10", "-3", "8.5", "Score: 8", "008"):
            output = weighted_output(completion, "8", [("8", 1.0)])
            with self.subTest(completion=completion), self.assertRaises(ValueError):
                extract_weighted_score(output)

    def test_malformed_sampled_token_is_not_repaired(self):
        output = weighted_output("10", "1_0", [("10", 1.0)])
        with self.assertRaisesRegex(ValueError, "does not match"):
            extract_weighted_score(output)

    def test_missing_and_inconsistent_sampled_probability_evidence_is_rejected(self):
        for options in ([(" 8", 1.0)], [("8", 0.25)]):
            output = weighted_output("8", "8", options)
            assert output.choices[0].logprobs
            output.choices[0].logprobs.content[0].logprob = 0.0
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "matching logprobs"),
            ):
                extract_weighted_score(output, 8, 8)

    def test_duplicate_tokens_and_invalid_probability_mass_are_rejected(self):
        output = weighted_output("8", "8", [("8", 0.5), ("8", 0.5)])
        with self.assertRaisesRegex(ValueError, "duplicate token"):
            extract_weighted_score(output, 8, 8)
        for logprob in (float("nan"), float("inf"), 0.1, -math.inf):
            output = weighted_output("8", "8", [("8", 1.0)])
            assert output.choices[0].logprobs
            token = output.choices[0].logprobs.content[0]
            token.logprob = logprob
            assert token.top_logprobs
            token.top_logprobs[0].logprob = logprob
            with self.subTest(logprob=logprob), self.assertRaises(ValueError):
                extract_weighted_score(output, 8, 8)

    def test_sampled_token_cannot_have_zero_probability(self):
        output = weighted_output("8", "8", [("8", 0.0), ("9", 1.0)])
        with self.assertRaisesRegex(ValueError, "finite non-positive"):
            extract_weighted_score(output, 8, 9)

    def test_distinct_valid_spellings_are_aggregated_without_double_counting(self):
        output = weighted_output("8", "8", [("8", 0.2), (" 8", 0.3), ("9", 0.5)])
        self.assertEqual(extract_weighted_score(output, 8, 9), 8.5)

    def test_valid_binary_weighting_retains_its_existing_arithmetic(self):
        output = weighted_output("true", "true", [("true", 0.6), ("false", 0.4)])
        self.assertAlmostEqual(extract_weighted_binary_answer(output), 0.6)
