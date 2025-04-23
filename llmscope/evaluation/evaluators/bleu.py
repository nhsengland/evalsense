from typing import Any, cast, override

import evaluate
from inspect_ai.scorer import (
    MetricProtocol,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import concurrency

from llmscope.evaluation import Evaluator, ScoreCalculator


class BleuPrecisionScoreCalculator(ScoreCalculator):
    """Calculator for computing BLEU scores."""

    def __init__(self):
        self.bleu_module = evaluate.load("bleu")

    @override
    def calculate(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        **kwargs: dict,
    ) -> Score:
        """
        Calculates BLEU precision scores for the supplied model prediction and reference input.

        Args:
            prediction (str): The text of the prediction from the model.
            reference (str, optional): The text of the reference input to compare against.

        Returns:
            Score: Inspect AI Score with the calculated evaluation results.
        """
        if reference is None:
            raise ValueError(
                "Reference is required for computing BLEU precision, but was None."
            )

        predictions = [prediction]
        references = [reference]

        result = self.bleu_module.compute(
            predictions=predictions, references=references
        )
        return Score(
            value=result["precisions"][0],  # type: ignore
            answer=prediction,
            metadata={
                "prediction": prediction,
                "reference": reference,
            },
        )

    @override
    async def calculate_async(
        self, *, prediction: str, reference: str | None = None, **kwargs: dict
    ) -> Score:
        """
        Calculates BLEU precision scores for the supplied model prediction and reference input.

        Args:
            prediction (str): The text of the prediction from the model.
            reference (str, optional): The text of the reference input to compare against.

        Returns:
            Score: Inspect AI Score with the calculated evaluation results.
        """
        return self.calculate(prediction=prediction, reference=reference, **kwargs)


_bleu_calculator: BleuPrecisionScoreCalculator | None = None


async def _init_bleu() -> BleuPrecisionScoreCalculator:
    """
    Lazily initialises the BLEU precision score calculator.

    Returns:
        BleuPrecisionScoreCalculator: The initialised BLEU calculator.
    """
    async with concurrency("load_bleu", 1):
        global _bleu_calculator
        if _bleu_calculator is None:
            _bleu_calculator = BleuPrecisionScoreCalculator()

    return _bleu_calculator


def bleu_base() -> MetricProtocol:
    """
    Base metric for BLEU scores.

    Returns:
        MetricProtocol: A function that computes BLEU scores.
    """

    def metric(scores: list[SampleScore]) -> Value:
        bleu_module = evaluate.load("bleu")
        predictions = [score.score.metadata["prediction"] for score in scores]  # type: ignore
        references = [score.score.metadata["reference"] for score in scores]  # type: ignore
        result = bleu_module.compute(predictions=predictions, references=references)
        result = cast(dict[str, Any], result)
        return result["bleu"]

    return metric


@metric(name="BLEU")
def bleu() -> MetricProtocol:
    """
    Metric for BLEU scores.

    Returns:
        MetricProtocol: A function that computes BLEU scores.
    """
    return bleu_base()


def bleu_precision_base() -> Scorer:
    """
    Base scorer for BLEU precision scores.

    Returns:
        Scorer: A coroutine that computes BLEU precision scores.
    """

    async def score(state: TaskState, target: Target):
        bleu_calculator = await _init_bleu()
        return await bleu_calculator.calculate_async(
            prediction=state.output.completion, reference=target.text
        )

    return score


@scorer(name="BLEU Precision", metrics=[bleu()])
def bleu_precision() -> Scorer:
    """
    Scorer for BLEU precision scores.

    Returns:
        Scorer: A coroutine that computes BLEU precision scores.
    """
    return bleu_precision_base()


bleu_evaluator = Evaluator("BLEU", scorer=bleu_precision())
