"""Record the estimator and probability evidence with each G-Eval score."""

from typing import cast
import unittest
from unittest.mock import AsyncMock, Mock

from inspect_ai.model import Model, ModelOutput

from evalsense.evaluation.evaluators.g_eval import (
    GEvalScoreCalculator,
    WEIGHTING_POLICY,
    ESTIMATOR_FORMAT,
)
from test_g_eval_weighted_ratings import weighted_output


class EstimatorMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_weighted_metadata_records_the_conditional_distribution(self):
        model = Mock(spec=Model)
        model.generate = AsyncMock(
            return_value=weighted_output(
                "8", "8", [("8", 0.1), (" 8", 0.15), ("9", 0.25), ("other", 0.5)]
            )
        )
        result = await GEvalScoreCalculator(
            cast(Model, model),
            "Rate {prediction}",
            min_score=8,
            max_score=9,
        ).calculate_async(prediction="candidate")
        assert result.metadata
        self.assertEqual(result.metadata["estimator_format"], ESTIMATOR_FORMAT)
        self.assertEqual(result.value, 0.5)
        self.assertEqual(result.metadata["raw_score"], 8.5)
        self.assertEqual(result.metadata["estimator"], "weighted_rating_tokens")
        self.assertEqual(result.metadata["weighting_policy"], WEIGHTING_POLICY)
        self.assertAlmostEqual(result.metadata["rating_probability_mass"], 0.5)
        self.assertEqual(result.metadata["rating_probabilities"], {"8": 0.5, "9": 0.5})
        self.assertEqual(result.metadata["output_text"], "8")

    async def test_text_only_scores_explicitly_record_the_sampled_estimator(self):
        model = Mock(spec=Model)
        model.generate = AsyncMock(
            return_value=ModelOutput.from_content("fixture", "010")
        )
        result = await GEvalScoreCalculator(
            cast(Model, model),
            "Rate {prediction}",
            logprobs=False,
            normalise=False,
        ).calculate_async(prediction="candidate")
        assert result.metadata
        self.assertEqual(result.metadata["estimator_format"], ESTIMATOR_FORMAT)
        self.assertEqual(result.value, 10)
        self.assertEqual(result.metadata["estimator"], "sampled_rating")
        self.assertNotIn("weighting_policy", result.metadata)
        self.assertNotIn("rating_probabilities", result.metadata)
        self.assertNotIn("rating_probability_mass", result.metadata)
