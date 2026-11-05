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
    """Legacy modified unigram precision calculator, without a brevity penalty.

    Retained for direct callers. The BLEU evaluator uses BleuScoreCalculator;
    unigram precision must not be interpreted as sentence or corpus BLEU.
    """

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
        Calculates modified unigram precision, not BLEU, for one text pair.

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
        Asynchronously call this calculator's calculate method with the same inputs.

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


class BleuScoreCalculator(BleuPrecisionScoreCalculator):
    """Unsmoothed sentence BLEU-4 for one complete prediction/reference pair.

    Uses Evaluate's BLEU backend and its default case-sensitive 13a tokenizer,
    uniform weights over n-gram orders 1 through 4, and the brevity penalty.
    ``Score.value`` is ``result["bleu"]``, not unigram precision. Scores are in
    [0, 1]. No smoothing or effective-order adjustment is applied: missing any
    n-gram order, including predictions shorter than four tokens, gives zero.
    Such short samples provide little information for correlation analyses.

    Corpus BLEU recomputes pooled n-gram counts and the corpus brevity penalty;
    it is not the mean of these sentence scores. Original texts are retained in
    score metadata for that calculation. Sync and async calls use the same policy.
    """

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
        """Compute the backend's BLEU-4 value for a single text pair.

        Args:
            prediction (str): Complete generated text.
            input (str | None): Unused source text.
            reference (str | None): Required reference text.
            metadata (dict[str, Any] | None): Unused sample metadata.
            **kwargs (Any): Unused arguments accepted by ScoreCalculator.

        Returns:
            Score: Sentence BLEU with input texts and the score definition recorded.

        Raises:
            ValueError: If a reference is missing or either text is blank.
                Backend errors propagate; no fallback score is returned.
        """
        if reference is None or not reference.strip() or not prediction.strip():
            raise ValueError(
                "Sentence BLEU requires non-empty prediction and reference texts."
            )
        result = self.bleu_module.compute(
            predictions=[prediction], references=[reference], max_order=4, smooth=False
        )
        result = cast(dict[str, Any], result)
        return Score(
            value=result["bleu"],
            answer=prediction,
            metadata={
                "prediction": prediction,
                "reference": reference,
                "bleu_sample_variant": "sentence_bleu_4",
                "max_order": 4,
                "smooth": False,
                "metric_provenance": self.implementation.provenance,
            },
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
        result = bleu_module.compute(
            predictions=predictions, references=references, max_order=4, smooth=False
        )
        result = cast(dict[str, Any], result)
        return result["bleu"]

    return metric


def get_bleu_evaluator(
    name: str = "BLEU",
    scorer_name: str = "Sentence BLEU",
    metrics: list[Metric | dict[str, list[Metric]]]
    | dict[str, list[Metric]]
    | None = None,
    *,
    version: str | None = None,
) -> Evaluator:
    """
    Return sentence BLEU-4 scores with a separately labelled corpus BLEU metric.

    Correlation and meta-evaluation consume unsmoothed sentence BLEU-4. The default
    aggregate is ``<name> (corpus)``, recomputed from the input texts; it is not a
    mean sentence score. Supply ``metrics=[mean()]`` to request the latter instead.
    See BleuScoreCalculator for tokenization and short-input behavior.

    Re-score under a fresh evaluator name or project when migrating from legacy
    unigram scores. Existing cached logs are not converted or invalidated here.

    Args:
        name (str): The evaluator name and prefix for the corpus metric. Defaults to "BLEU".
        scorer_name (str): The name of the internal scorer. Defaults to "Sentence BLEU".
        metrics (list[Metric | dict[str, list[Metric]]] | dict[str, list[Metric]] | None):
            The metrics to use for the evaluation. If `None`, the default metric
            will be used (unsmoothed corpus BLEU-4).

        version (str | None): Cache version for custom code or external resources.

    Returns:
        Evaluator: An evaluator for BLEU scores.
    """
    cache_parameters = {
        "method": "bleu",
        "scorer_name": scorer_name,
        "metrics": metrics,
    }

    bleu_calculator = BleuScoreCalculator()

    @metric(name=f"{name} (corpus)")
    def bleu() -> MetricProtocol:
        return bleu_metric(implementation=bleu_calculator.implementation)

    if metrics is None:
        metrics = [bleu()]

    @scorer(name=scorer_name, metrics=metrics)
    def sentence_bleu_scorer() -> Scorer:
        async def score(state: TaskState, target: Target):
            return await bleu_calculator.calculate_async(
                prediction=state.output.completion, reference=target.text
            )

        return score

    return Evaluator(
        name,
        version=version,
        cache_parameters=cache_parameters,
        scorer=sentence_bleu_scorer(),
        implementation=bleu_calculator.implementation,
    )
