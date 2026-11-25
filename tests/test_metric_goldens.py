"""Numerical regression fixtures; expected values are calculated independently."""

import asyncio
import math
import unittest
from unittest.mock import Mock, patch

from inspect_ai.scorer import SampleScore

from evalsense.evaluation.evaluators import (
    BleuPrecisionScoreCalculator,
    RougeScoreCalculator,
    bleu_metric,
)
from metric_fixtures import METRICS, metric_backend


class MetricGoldenTests(unittest.TestCase):
    def test_rouge_overlap_and_lcs_have_distinct_values(self):
        # Reference a b c d; prediction a c b x. Three unigram matches, no
        # bigrams, LCS length two. Equal lengths give F1 = 3/4, 0, 1/2.
        with metric_backend("rouge"):
            result = RougeScoreCalculator().calculate(
                prediction="a c b x", reference="a b c d"
            )
        self.assertEqual(
            result.value, {"ROUGE-1": 0.75, "ROUGE-2": 0.0, "ROUGE-L": 0.5}
        )

    def test_rouge_fmeasure_uses_both_lengths(self):
        # a b against a b c d: P1=1, R1=1/2; P2=1, R2=1/3.
        with metric_backend("rouge"):
            result = RougeScoreCalculator().calculate(
                prediction="a b", reference="a b c d"
            )
        self.assertEqual(
            result.value, {"ROUGE-1": 2 / 3, "ROUGE-2": 0.5, "ROUGE-L": 2 / 3}
        )

    def test_rouge_clips_repeated_words(self):
        # a a a a against a b c d: one matched unigram, not four.
        with metric_backend("rouge"):
            result = RougeScoreCalculator().calculate(
                prediction="a a a a", reference="a b c d"
            )
        self.assertEqual(
            result.value, {"ROUGE-1": 0.25, "ROUGE-2": 0.0, "ROUGE-L": 0.25}
        )

    def test_rouge_sync_and_async_preserve_identical_values(self):
        with metric_backend("rouge"):
            calculator = RougeScoreCalculator()
            sync = calculator.calculate(prediction="a b", reference="a b c d")
            async_result = asyncio.run(
                calculator.calculate_async(prediction="a b", reference="a b c d")
            )
        self.assertEqual(sync.model_dump(), async_result.model_dump())

    def test_corpus_bleu_pools_ngram_counts(self):
        with metric_backend("bleu"):
            calculator = BleuPrecisionScoreCalculator()
            scores = [
                SampleScore(
                    score=calculator.calculate(prediction=p, reference="a b c d")
                )
                for p in ("a b c d", "a b x y")
            ]
            value = bleu_metric()(scores)
        # P1=6/8, P2=4/6, P3=2/4, P4=1/2; total lengths equal, BP=1.
        assert isinstance(value, (int, float))
        self.assertAlmostEqual(float(value), (1 / 8) ** 0.25, places=12)

    def test_corpus_bleu_applies_length_penalty_after_pooling(self):
        with metric_backend("bleu"):
            calculator = BleuPrecisionScoreCalculator()
            scores = [
                SampleScore(
                    score=calculator.calculate(prediction="a b c d", reference=r)
                )
                for r in ("a b c d", "a b c d e f g h")
            ]
            value = bleu_metric()(scores)
        # All precisions are one; total lengths are 8 and 12.
        assert isinstance(value, (int, float))
        self.assertAlmostEqual(float(value), math.exp(-0.5), places=12)

    def test_backend_failure_cannot_become_a_metric_value(self):
        backend = Mock()
        backend.compute.side_effect = RuntimeError("synthetic backend failure")
        with patch("evaluate.load", return_value=backend):
            # The real-backend tests above cover arithmetic; this fixture covers
            # the wrapper's failure path only.
            for cls in (RougeScoreCalculator, BleuPrecisionScoreCalculator):
                with self.subTest(calculator=cls.__name__):
                    # Pinned-loader validation, when present, needs the real hash.
                    name = "rouge" if cls is RougeScoreCalculator else "bleu"

                    backend._hash = METRICS[name][1]
                    with self.assertRaisesRegex(RuntimeError, "synthetic backend"):
                        cls().calculate(prediction="a b c d", reference="a b c d")


if __name__ == "__main__":
    unittest.main()
