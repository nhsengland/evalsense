"""A rating response must be one complete integer, never a numeric substring."""

import unittest

from evalsense.utils.text import extract_score


class ScoreResponseTests(unittest.TestCase):
    def test_accepts_complete_in_range_integers_with_outer_whitespace(self):
        for text, expected in (("1", 1), ("10", 10), ("\t8\n", 8), ("008", 8)):
            with self.subTest(text=text):
                self.assertEqual(extract_score(text), expected)
        self.assertEqual(extract_score("0", 0, 5), 0)

    def test_rejects_the_reported_score_inversions(self):
        for text in ("0/10", "-3", "8.5", "Rating scale 1 to 10. Final rating 9"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract_score(text)

    def test_rejects_wrappers_prose_and_multiple_ratings(self):
        for text in (
            "",
            " ",
            "Score: 8",
            "8 out of 10",
            "8/10",
            "8\n9",
            "8 because correct",
            '{"score": 8}',
            '"8"',
            "**8**",
            "```\n8\n```",
            "(8)",
            "8.",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract_score(text)

    def test_rejects_alternative_number_syntax(self):
        for text in ("+8", "8.0", "1e0", "0x8", "1_0", "1 0", "８", "٨", "\x008"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract_score(text)

    def test_rejects_out_of_range_values_without_echoing_the_response(self):
        for text in ("0", "11", "secret clinical text 0/10", "9" * 5000):
            with self.subTest(length=len(text)), self.assertRaises(ValueError) as error:
                extract_score(text)
            self.assertNotIn(text, str(error.exception))
