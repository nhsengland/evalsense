from typing import Any, cast, override

from inspect_ai.scorer import (
    Metric,
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

from evalsense.evaluation import Evaluator, ScoreCalculator
from evalsense.evaluation.metric_implementation import MetricImplementation


class BleuPrecisionScoreCalculator(ScoreCalculator):
    """Calculator for computing BLEU scores."""

    def __init__(self):
        self.implementation = MetricImplementation.resolve("bleu")
        self.bleu_module = self.implementation.load()

    @override
    def calculate(
        self,
        *,
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> Score:
        """
        Calculates BLEU precision scores for the supplied model prediction and reference input.

        Args:
            prediction (str): The text of the prediction from the model.
            input (str, optional): The text of the input to the model. Ignored for BLEU.
            reference (str, optional): The text of the reference input to compare against.
            metadata (dict[str, Any], optional): Additional metadata for the score.
                Ignored for BLEU.

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
                "metric_provenance": self.implementation.provenance,
            },
        )

    @override
    async def calculate_async(
        self,
        *,
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> Score:
        """
        Calculates BLEU precision scores for the supplied model prediction and reference input.

        Args:
            prediction (str): The text of the prediction from the model.
            input (str, optional): The text of the input to the model. Ignored for BLEU.
            reference (str, optional): The text of the reference input to compare against.
            metadata (dict[str, Any], optional): Additional metadata for the score.
                Ignored for BLEU.

        Returns:
            Score: Inspect AI Score with the calculated evaluation results.
        """
        return self.calculate(
            prediction=prediction,
            reference=reference,
            input=input,
            metadata=metadata,
            **kwargs,
        )


def bleu_metric(
    *, implementation: MetricImplementation | None = None
) -> MetricProtocol:
    """
    Base metric for BLEU scores.

    Args:
        implementation (MetricImplementation | None): Pinned implementation shared
            with the sample scorer. Defaults to the supported BLEU implementation.

    Rejects input scores with absent or different implementation provenance.
    Historical scores must be rescored before recomputing this corpus metric.

    Returns:
        MetricProtocol: A function that computes BLEU scores.
    """

    implementation = implementation or MetricImplementation.resolve("bleu")

    def metric(scores: list[SampleScore]) -> Value:
        if any(
            (score.score.metadata or {}).get("metric_provenance")
            != implementation.provenance
            for score in scores
        ):
            raise ValueError(
                "Corpus BLEU requires scores from the same recorded metric implementation."
            )
        bleu_module = implementation.load()
        predictions = [score.score.metadata["prediction"] for score in scores]  # type: ignore
        references = [score.score.metadata["reference"] for score in scores]  # type: ignore
        result = bleu_module.compute(predictions=predictions, references=references)
        result = cast(dict[str, Any], result)
        return result["bleu"]

    return metric


def get_bleu_evaluator(
    name: str = "BLEU",
    scorer_name: str = "BLEU Precision",
    metrics: list[Metric | dict[str, list[Metric]]]
    | dict[str, list[Metric]]
    | None = None,
) -> Evaluator:
    """
    Returns an evaluator for BLEU scores.

    Args:
        name (str): The name of the metric and evaluator. Defaults to "BLEU".
        scorer_name (str): The name of the internal scorer. Defaults to "BLEU Precision".
        metrics (list[Metric | dict[str, list[Metric]]] | dict[str, list[Metric]] | None):
            The metrics to use for the evaluation. If `None`, the default metric
            will be used (BLEU).

    Returns:
        Evaluator: An evaluator for BLEU scores.
    """

    bleu_calculator = BleuPrecisionScoreCalculator()

    @metric(name=name)
    def bleu() -> MetricProtocol:
        return bleu_metric(implementation=bleu_calculator.implementation)

    if metrics is None:
        metrics = [bleu()]

    @scorer(name=scorer_name, metrics=metrics)
    def bleu_precision_scorer() -> Scorer:
        async def score(state: TaskState, target: Target):
            return await bleu_calculator.calculate_async(
                prediction=state.output.completion, reference=target.text
            )

        return score

    return Evaluator(
        name,
        scorer=bleu_precision_scorer(),
        implementation=bleu_calculator.implementation,
    )
