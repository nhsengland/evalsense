from typing import Any, cast

import evaluate
from inspect_ai.scorer import (
    MetricProtocol,
    SampleScore,
    Score,
    Target,
    Value,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import concurrency

from llmscope.evaluation import Evaluator

_bleu_fun: evaluate.EvaluationModule | None = None


async def _load_bleu():
    async with concurrency("load_bleu", 1):
        global _bleu_fun
        if _bleu_fun is None:
            _bleu_fun = evaluate.load("bleu")

    return _bleu_fun


def bleu_base() -> MetricProtocol:
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
    return bleu_base()


def bleu_precision_base():
    async def score(state: TaskState, target: Target):
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
def bleu_precision():
    return bleu_precision_base()


bleu_evaluator = Evaluator("BLEU", scorer=bleu_precision())
