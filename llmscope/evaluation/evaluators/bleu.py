from typing import Any, cast

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

from llmscope.evaluation import Evaluator

_bleu_fun: evaluate.EvaluationModule | None = None


async def _load_bleu() -> evaluate.EvaluationModule:
    """
    Lazily loads the BLEU evaluation module.

    Returns:
        evaluate.EvaluationModule: The loaded BLEU evaluation module.
    """
    async with concurrency("load_bleu", 1):
        global _bleu_fun
        if _bleu_fun is None:
            _bleu_fun = evaluate.load("bleu")

    return _bleu_fun


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
        if not target.text:
            raise ValueError("Non-empty target is required for BLEU evaluation.")

        bleu_module = await _load_bleu()
        predictions = [state.output.completion]
        references = [target.text]
        result = bleu_module.compute(predictions=predictions, references=references)
        return Score(
            value=result["precisions"][0],  # type: ignore
            answer=state.output.completion,
            metadata={
                "prediction": state.output.completion,
                "reference": target.text,
            },
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
