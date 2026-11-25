"""Scoring-policy changes must not reuse results from the permissive parser."""

from dataclasses import replace
from typing import cast
import unittest
from unittest.mock import AsyncMock, Mock

from inspect_ai.model import Model, ModelOutput

from evalsense.evaluation.evaluators.g_eval import (
    GEvalScoreCalculator,
    SCORE_FORMAT,
    ESTIMATOR_FORMAT,
    WEIGHTING_POLICY,
    get_g_eval_evaluator,
)
from evalsense.generation import ModelConfig
from test_cache_identity import experiment


def evaluator():
    return get_g_eval_evaluator(
        prompt_template="Rate {prediction}",
        model_config=ModelConfig("mockllm/judge"),
        logprobs=False,
    )


class RatingPolicyTests(unittest.TestCase):
    def test_parser_policy_invalidates_scores_but_preserves_generation_identity(self):
        current = evaluator()
        assert current.cache_parameters is not None
        self.assertEqual(current.cache_parameters["score_format"], SCORE_FORMAT)
        previous = replace(
            current,
            cache_parameters={
                key: value
                for key, value in current.cache_parameters.items()
                if key != "score_format"
            },
        )
        old, new = experiment(evaluator=previous), experiment(evaluator=current)
        self.assertEqual(old.generation_record, new.generation_record)
        self.assertNotEqual(old.evaluation_record, new.evaluation_record)
        self.assertNotEqual(previous.cache_key, current.cache_key)
        self.assertEqual(current.cache_key, evaluator().cache_key)

    def test_weighted_policy_does_not_reuse_legacy_weighting_results(self):
        current = get_g_eval_evaluator(
            prompt_template="Rate {prediction}",
            model_config=ModelConfig("mockllm/judge"),
            logprobs=True,
        )
        assert current.cache_parameters
        self.assertEqual(current.cache_parameters["weighting_policy"], WEIGHTING_POLICY)
        legacy = replace(
            current,
            cache_parameters={
                key: value
                for key, value in current.cache_parameters.items()
                if key != "weighting_policy"
            },
        )
        self.assertNotEqual(current.cache_key, legacy.cache_key)
        self.assertEqual(
            experiment(evaluator=current).generation_record,
            experiment(evaluator=legacy).generation_record,
        )
        self.assertNotIn("weighting_policy", evaluator().cache_parameters or {})

    def test_estimator_format_requires_new_scores_without_regenerating_outputs(self):
        for logprobs in (False, True):
            current = get_g_eval_evaluator(
                prompt_template="Rate {prediction}",
                model_config=ModelConfig("mockllm/judge"),
                logprobs=logprobs,
            )
            assert current.cache_parameters
            self.assertEqual(
                current.cache_parameters["estimator_format"], ESTIMATOR_FORMAT
            )
            previous = replace(
                current,
                cache_parameters={
                    key: value
                    for key, value in current.cache_parameters.items()
                    if key != "estimator_format"
                },
            )
            with self.subTest(logprobs=logprobs):
                self.assertNotEqual(current.cache_key, previous.cache_key)
                old, new = experiment(evaluator=previous), experiment(evaluator=current)
                self.assertEqual(old.generation_record, new.generation_record)
                self.assertNotEqual(old.evaluation_record, new.evaluation_record)


class RatingPolicyMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_score_records_the_parser_protocol(self):
        model = Mock(spec=Model)
        model.generate = AsyncMock(
            return_value=ModelOutput.from_content("fixture", "8")
        )
        result = await GEvalScoreCalculator(
            cast(Model, model), "Rate {prediction}", logprobs=False
        ).calculate_async(prediction="candidate")
        assert result.metadata
        self.assertEqual(result.metadata["score_format"], SCORE_FORMAT)
        self.assertEqual(result.metadata["output_text"], "8")
        self.assertEqual(result.metadata["raw_score"], 8)
        self.assertAlmostEqual(cast(float, result.value), 7 / 9)
