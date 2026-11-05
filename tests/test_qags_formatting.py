"""Formatting tolerance must not change the meaning of questions or answers."""

import itertools
import unittest
from typing import Any, cast
from unittest.mock import patch

from evalsense.evaluation.evaluators.qags import QagsValidationError
from test_qags import calculator, model_with, ternary


class AnswerFormattingTests(unittest.TestCase):
    def test_balanced_wrappers_preserve_the_ternary_truth_table(self):
        wrappers = [
            ("", ""),
            ('"', '"'),
            ("'", "'"),
            ("“", "”"),
            ("‘", "’"),
            ("*", "*"),
            ("**", "**"),
            ("_", "_"),
            ("__", "__"),
            ("`", "`"),
        ]
        for pa, ra in itertools.product(("yes", "no", "unknown"), repeat=2):
            expected = ternary([pa], [ra]).value
            for opening, closing in wrappers:
                with self.subTest(pa=pa, ra=ra, wrapper=opening):
                    result = ternary(
                        [f"{opening}{pa}{closing}"], [f"{opening}{ra}{closing}"]
                    )
                    self.assertEqual(result.value, expected)

    def test_whitespace_case_and_single_punctuation_are_accepted(self):
        for text in [
            " ** Yes ** ",
            '"YES."',
            '"yes"!',
            "`True!`",
            "_ true _.",
            "‘I don’t know’.",
            "**unknown!**",
            "'I don't know'",
            "“no”",
        ]:
            with self.subTest(text=text):
                result = ternary([text], [text])
                assert isinstance(result.value, dict)
                self.assertEqual(result.value["fixture Accuracy"], 1.0)

    def test_explanations_contradictions_and_malformed_wrappers_fail(self):
        invalid = [
            "Yes, the text says so.",
            "Yes, but actually no.",
            "not true",
            "No evidence of disease",
            '"yes" or "no"',
            "**yes or no**",
            "**yes*",
            '"yes',
            'yes"',
            "'yes\"",
            '"yes.".',
            "yes!!",
            '**"yes"**',
            "```yes```",
            "~~yes~~",
            "[yes](https://example.org)",
            '{"answer":"yes"}',
            "**unknown cause**",
            "*no response received*",
            "answer: yes",
            '""',
            "** **",
            "`no`\nyes",
            '"no\u200b"',
        ]
        for text, side in itertools.product(invalid, ("prediction", "reference")):
            with (
                self.subTest(text=text, side=side),
                self.assertRaises(QagsValidationError),
            ):
                ternary(
                    [text if side == "prediction" else "yes"],
                    [text if side == "reference" else "yes"],
                )

    def test_generic_helpers_keep_legacy_behavior_separate(self):
        from evalsense.utils.text import (
            extract_lines,
            extract_ternary_answer,
            parse_ternary_answer,
        )

        self.assertIsNone(extract_ternary_answer("unparseable", binary_only=False))
        self.assertEqual(extract_lines("1.Question?"), ["Question?"])
        with self.assertRaises(ValueError):
            parse_ternary_answer("unparseable")
        self.assertIs(parse_ternary_answer('"YES"'), True)
        self.assertIs(parse_ternary_answer("**no**", binary_only=True), False)
        self.assertIsNone(parse_ternary_answer("‘I don’t know’"))
        for text in ("unknown", '"unknown"', "**I don’t know**"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_ternary_answer(text, binary_only=True)

    def test_generic_error_never_echoes_source_text(self):
        from evalsense.utils.text import parse_ternary_answer

        with self.assertRaises(ValueError) as caught:
            parse_ternary_answer("confidential clinical text")
        self.assertNotIn("confidential clinical text", str(caught.exception))
        with self.assertRaises(QagsValidationError) as wrapped:
            ternary(["confidential clinical text"], ["yes"])
        self.assertNotIn("confidential clinical text", str(wrapped.exception))
        self.assertIn("prediction", str(wrapped.exception))


class QuestionFormattingTests(unittest.IsolatedAsyncioTestCase):
    async def test_unspaced_prefixes_deduplicate_the_same_question(self):
        for prefix in ("1.", "2)", "[3]", "-", "*", "•", "2. 3)", "• 1."):
            with self.subTest(prefix=prefix):
                fake = model_with(
                    prefix + "Does it work?", "Does it work?", "yes", "yes"
                )
                result = await calculator(model=fake).calculate_async(
                    prediction="candidate"
                )
                assert result.metadata is not None
                self.assertEqual(result.metadata["questions"], ["Does it work?"])
                self.assertEqual(
                    result.metadata["question_sources"], [["prediction", "reference"]]
                )
                self.assertEqual(
                    result.metadata["question_filtering"]["duplicate_questions"], 1
                )
                self.assertEqual(cast(Any, fake.generate).await_count, 4)

    async def test_decimal_and_signed_quantities_remain_distinct(self):
        originals = [
            "1.5 mg was prescribed?",
            "5 mg was prescribed?",
            "-2 mg was prescribed?",
            "2 mg was prescribed?",
            "-.5 mg was prescribed?",
            ".5 mg was prescribed?",
            "1.25 mg was prescribed?",
            "25 mg was prescribed?",
            "-2.5 mg was prescribed?",
            "+2.5 mg was prescribed?",
        ]
        fake = model_with("\n".join(originals), "")
        meta: dict[str, Any] = {}
        result = await calculator(model=fake)._generate_questions(
            prediction="candidate", score_metadata=meta
        )
        self.assertEqual(result, originals)
        self.assertEqual(meta["question_filtering"]["duplicate_questions"], 0)

    async def test_ambiguous_numeric_prefix_is_preserved(self):
        from evalsense.utils.text import strip_list_prefix

        for text in (
            "1.5 mg?",
            "-2 mg?",
            "-.5 mg?",
            "1)2 mg?",
            "[1]2 mg?",
            "*2 mg?",
            "1.2.Does it work?",
            "-² mg?",
            "1.² mg?",
            "-٢ mg?",
            "-Ⅳ mg?",
        ):
            with self.subTest(text=text):
                self.assertEqual(strip_list_prefix(text), text)
        for text in ("1. -2 mg?", "- 1.5 mg?", "[1] 2 mg?"):
            expected = {
                "1. -2 mg?": "-2 mg?",
                "- 1.5 mg?": "1.5 mg?",
                "[1] 2 mg?": "2 mg?",
            }[text]
            self.assertEqual(strip_list_prefix(text), expected)

    async def test_prefix_normalization_preserves_case_unicode_and_quoted_spaces(self):
        originals = [
            "Is the unit mM?",
            "Is the unit mm?",
            'Does "a  b" appear?',
            'Does "a b" appear?',
            "Est-ce mentionné?",
            "Ist es erwähnt?",
        ]
        fake = model_with("\n".join("1." + item for item in originals), "")
        actual = await calculator(model=fake)._generate_questions(
            prediction="candidate", score_metadata={}
        )
        self.assertEqual(actual, originals)

    async def test_formatted_answers_work_through_public_calculator(self):
        fake = model_with("1.Does it work?", "[2]Does it work?", "**Yes**", '"yes"')
        score = await calculator(model=fake).calculate_async(prediction="candidate")
        assert score.metadata is not None
        self.assertEqual(score.metadata["raw_prediction_answers"], ["**Yes**"])
        self.assertEqual(score.metadata["raw_reference_answers"], ['"yes"'])
        self.assertEqual(score.metadata["prediction_answers"], [True])
        assert isinstance(score.value, dict)
        self.assertEqual(score.value["fixture Accuracy"], 1.0)
        self.assertEqual(cast(Any, fake.generate).await_count, 4)

    async def test_judge_formatting_does_not_change_weighted_scoring(self):
        for answer in ("**Yes**", '"no"', "`true`"):
            with self.subTest(answer=answer):
                fake = model_with(answer)
                with patch(
                    "evalsense.evaluation.evaluators.qags.extract_weighted_binary_answer",
                    return_value=0.25,
                ) as weighted:
                    score = await calculator(
                        "judge", fake, True
                    )._evaluate_judge_answers(
                        prediction="candidate",
                        questions=["One?"],
                        raw_prediction_answers=["alpha"],
                        raw_reference_answers=["beta"],
                        score_metadata={},
                    )
                self.assertEqual(score.value, 0.25)
                self.assertEqual(weighted.call_args.args[0].completion, answer)

    async def test_judge_rejects_wrapped_unknown_and_conflicting_labels(self):
        for answer in (
            "**unknown**",
            '"I don’t know"',
            "**yes** or **no**",
            "Yes, the text says so.",
        ):
            with self.subTest(answer=answer):
                with patch(
                    "evalsense.evaluation.evaluators.qags.extract_weighted_binary_answer"
                ) as weighted:
                    with self.assertRaises(QagsValidationError):
                        await calculator(
                            "judge", model_with(answer), True
                        )._evaluate_judge_answers(
                            prediction="candidate",
                            questions=["One?"],
                            raw_prediction_answers=["alpha"],
                            raw_reference_answers=["beta"],
                            score_metadata={},
                        )
                    weighted.assert_not_called()

    async def test_invalid_answer_is_not_retried_or_removed(self):
        fake = model_with("One?", "One?", "Yes, but no.", "yes")
        with self.assertRaises(QagsValidationError):
            await calculator(model=fake).calculate_async(prediction="candidate")
        self.assertEqual(cast(Any, fake.generate).await_count, 4)

    async def test_exact_mode_keeps_its_existing_text_comparison(self):
        score = calculator("exact")._evaluate_exact_answers(
            prediction="candidate",
            questions=["One?"],
            raw_prediction_answers=['"yes"'],
            raw_reference_answers=["yes"],
            score_metadata={},
        )
        self.assertEqual(score.value, 0.0)


if __name__ == "__main__":
    unittest.main()
