import evaluate
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    mean,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import concurrency

from llmscope.evaluation import Evaluator

_rouge_fun: evaluate.EvaluationModule | None = None


async def _load_rouge() -> evaluate.EvaluationModule:
    """
    Lazily loads the ROUGE evaluation module.

    Returns:
        evaluate.EvaluationModule: The loaded ROUGE evaluation module.
    """
    async with concurrency("load_rouge", 1):
        global _rouge_fun
        if _rouge_fun is None:
            _rouge_fun = evaluate.load("rouge")

    return _rouge_fun


def rouge_base() -> Scorer:
    """
    Base scorer for ROUGE scores.

    Returns:
        Scorer: A coroutine that computes ROUGE scores.
    """

    async def score(state: TaskState, target: Target) -> Score:
        if not target.text:
            raise ValueError("Non-empty target is required for ROUGE evaluation.")

        rouge_module = await _load_rouge()
        predictions = [state.output.completion]
        references = [target.text]
        result = rouge_module.compute(predictions=predictions, references=references)
        return Score(
            value={
                "ROUGE-1": result["rouge1"],  # type: ignore
                "ROUGE-2": result["rouge2"],  # type: ignore
                "ROUGE-L": result["rougeL"],  # type: ignore
            },
            answer=state.output.completion,
        )

    return score


@scorer(
    name="ROUGE",
    metrics=[
        {
            "ROUGE-1": [mean()],
            "ROUGE-2": [mean()],
            "ROUGE-L": [mean()],
        }
    ],
)
def rouge() -> Scorer:
    """
    Scorer for ROUGE scores.

    Returns:
        Scorer: A coroutine that computes ROUGE scores.
    """
    return rouge_base()


rouge_evaluator = Evaluator("ROUGE", scorer=rouge())
