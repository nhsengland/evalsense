from typing import Any, override

import evaluate
from inspect_ai.scorer import (
    Metric,
    Score,
    Scorer,
    Target,
    mean,
    scorer,
)
from inspect_ai.solver import TaskState

from evalsense.evaluation import Evaluator, ScoreCalculator


class RougeScoreCalculator(ScoreCalculator):
    """Calculator for computing ROUGE scores."""

    def __init__(self):
        self.rouge_module = evaluate.load("rouge")

    @override
    def calculate(
        self,
        *,
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: dict,
    ) -> Score:
        """
        Calculates ROUGE scores for the supplied model prediction and reference input.

        Args:
            prediction (str): The text of the prediction from the model.
            input (str, optional): The text of the input to the model. Ignored for ROUGE.
            reference (str, optional): The text of the reference input to compare against.
            metadata (dict[str, Any], optional): Additional metadata for the score.
                Ignored for ROUGE.

        Returns:
            Score: Inspect AI Score with the calculated evaluation results.
        """
        if reference is None:
            raise ValueError("Reference is required for computing ROUGE, but was None.")

        predictions = [prediction]
        references = [reference]

        result = self.rouge_module.compute(
            predictions=predictions, references=references
        )
        return Score(
            value={
                "ROUGE-1": result["rouge1"],  # type: ignore
                "ROUGE-2": result["rouge2"],  # type: ignore
                "ROUGE-L": result["rougeL"],  # type: ignore
            },
            answer=prediction,
        )

    @override
    async def calculate_async(
        self,
        *,
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: dict,
    ) -> Score:
        """
        Calculates ROUGE scores for the supplied model prediction and reference input.

        Args:
            prediction (str): The text of the prediction from the model.
            input (str, optional): The text of the input to the model. Ignored for ROUGE.
            reference (str, optional): The text of the reference input to compare against.
            metadata (dict[str, Any], optional): Additional metadata for the score.
                Ignored for ROUGE.

        Returns:
            Score: Inspect AI Score with the calculated evaluation results.
        """
        return self.calculate(
            prediction=prediction,
            input=input,
            reference=reference,
            metadata=metadata,
            **kwargs,
        )


def get_rouge_evaluator(
    name: str = "ROUGE",
    metrics: list[Metric | dict[str, list[Metric]]]
    | dict[str, list[Metric]]
    | None = None,
) -> Evaluator:
    """
    Returns an evaluator for ROUGE scores.

    Args:
        name (str): The name of the evaluator. Defaults to "ROUGE".
        metrics (list[Metric | dict[str, list[Metric]]] | dict[str, list[Metric]] | None):
            The metrics to use for evaluation. If None, defaults to ROUGE-1, ROUGE-2,
            and ROUGE-L with mean aggregation.

    Returns:
        Evaluator: An evaluator for ROUGE scores.
    """
    if metrics is None:
        metrics = [
            {
                "ROUGE-1": [mean()],
                "ROUGE-2": [mean()],
                "ROUGE-L": [mean()],
            }
        ]

    rouge_calculator = RougeScoreCalculator()

    @scorer(name=name, metrics=metrics)
    def rouge_scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return await rouge_calculator.calculate_async(
                prediction=state.output.completion, reference=target.text
            )

        return score

    return Evaluator(name, scorer=rouge_scorer())
