from typing import Any, override

from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import (
    Metric,
    Score,
    Scorer,
    Target,
    mean,
    scorer,
)
from inspect_ai.solver import TaskState

from evalsense.evaluation import (
    Evaluator,
    ScoreCalculator,
    ScorerFactory,
)
from evalsense.generation import ModelConfig
from evalsense.utils.text import (
    extract_score,
    extract_weighted_rating,
    format_template,
    validate_score_range,
)

SCORE_FORMAT = "evalsense.g-eval.integer.v1"
WEIGHTING_POLICY = "evalsense.g-eval.single-token.v1"
ESTIMATOR_FORMAT = "evalsense.g-eval.estimator.v1"


def validate_weighting_settings(
    logprobs: bool, top_logprobs: int, min_score: int, max_score: int
) -> None:
    """Reject a weighting request that cannot cover the configured rating scale."""
    if type(logprobs) is not bool:
        raise ValueError("logprobs must be a boolean.")
    if logprobs:
        if type(top_logprobs) is not int or top_logprobs < 1:
            raise ValueError(
                "Weighted ratings require top_logprobs to be a positive integer."
            )
        if top_logprobs < max_score - min_score + 1:
            raise ValueError(
                "top_logprobs must cover every rating on the configured scale."
            )


class GEvalScoreCalculator(ScoreCalculator):
    """G-Eval with complete, in-range integer responses.

    Weighted mode requires one output token and complete rating evidence in the
    returned alternatives. Failures never fall back to a sampled rating.
    Syntax validation does not establish judge accuracy or injection resistance.
    """

    def __init__(
        self,
        model: Model,
        prompt_template: str,
        logprobs: bool = True,
        top_logprobs: int = 20,
        min_score: int = 1,
        max_score: int = 10,
        normalise: bool = True,
        debug: bool = False,
    ):
        """
        Initializes the G-Eval score calculator.

        Args:
            model (Model): The model to use for evaluation.
            prompt_template (str): The prompt template with the scoring instructions.
            logprobs (bool): Use a one-token conditional mean over reported valid
                rating alternatives. Every allowed rating must be present; missing
                or inconsistent evidence raises ValueError. False uses the sampled
                integer without a one-token limit.
            top_logprobs (int): Positive integer alternative count, at least the
                number of allowed ratings in weighted mode. Providers may impose
                their own limit or omit low-probability ratings.
            min_score (int): The minimum valid score.
            max_score (int): The maximum valid score.
            normalise (bool): Whether to normalise the scores between 0 and 1.
            debug (bool): Retained for compatibility. Weighting failures always raise.
        """
        validate_score_range(min_score, max_score, normalise=normalise)
        validate_weighting_settings(logprobs, top_logprobs, min_score, max_score)
        self.model = model
        self.prompt_template = prompt_template
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.min_score = min_score
        self.max_score = max_score
        self.normalise = normalise
        self.debug = debug

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
        """This method is not supported for G-Eval and will raise an error when called.

        Use `calculate_async` instead.

        Raises:
            NotImplementedError: When called, as synchronous evaluation is not
                supported for G-Eval.
        """
        raise NotImplementedError(
            "Synchronous evaluation is not supported for G-Eval. "
            "Use calculate_async instead."
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
        """Calculates the G-Eval score asynchronously.

        Args:
            prediction (str): The predicted output to evaluate.
            input (str | None): The input text for the model. Defaults to `None`.
            reference (str | None): The reference text for the model. Defaults to `None`.
            metadata (dict[str, Any] | None): Additional metadata for the evaluation.
                Defaults to `None`.
            **kwargs: Additional keyword arguments.

        Returns:
            Score: The calculated score.

        Raises:
            ValueError: If settings, the complete integer response or weighted
                token evidence are invalid. Failed weighting never returns a
                sampled score, and malformed responses are never repaired.
        """
        validate_score_range(self.min_score, self.max_score, normalise=self.normalise)
        validate_weighting_settings(
            self.logprobs, self.top_logprobs, self.min_score, self.max_score
        )
        logprobs_config = GenerateConfig(
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs if self.logprobs else None,
            max_tokens=1 if self.logprobs else None,
            num_choices=1 if self.logprobs else None,
        )
        if metadata is None:
            metadata = {}
        llm_input = format_template(
            self.prompt_template,
            prediction=prediction,
            reference=reference,
            input=input,
            **metadata,
        )
        output = await self.model.generate(llm_input, config=logprobs_config)

        raw_score = extract_score(output.completion, self.min_score, self.max_score)
        weighted_rating = None
        if self.logprobs:
            weighted_rating = extract_weighted_rating(
                output, min_score=self.min_score, max_score=self.max_score
            )
            raw_score = weighted_rating.score

        if self.normalise:
            score = (raw_score - self.min_score) / (self.max_score - self.min_score)
        else:
            score = raw_score

        score_metadata: dict[str, Any] = {
            "prompt": llm_input,
            "output_text": output.completion,
            "raw_score": raw_score,
            "score_format": SCORE_FORMAT,
            "estimator_format": ESTIMATOR_FORMAT,
            "estimator": "weighted_rating_tokens"
            if self.logprobs
            else "sampled_rating",
        }
        if weighted_rating is not None:
            score_metadata.update(
                weighting_policy=WEIGHTING_POLICY,
                rating_probabilities={
                    str(rating): probability
                    for rating, probability in sorted(
                        weighted_rating.probabilities.items()
                    )
                },
                rating_probability_mass=weighted_rating.probability_mass,
            )
        return Score(value=score, answer=prediction, metadata=score_metadata)


class GEvalScorerFactory(ScorerFactory):
    """Scorer factory for G-Eval."""

    def __init__(
        self,
        name: str,
        prompt_template: str,
        metrics: list[Metric | dict[str, list[Metric]]]
        | dict[str, list[Metric]]
        | None = None,
        logprobs: bool = True,
        top_logprobs: int = 20,
        min_score: int = 1,
        max_score: int = 10,
        normalise: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the G-Eval scorer factory.

        Args:
            name (str): The name of the scorer.
            prompt_template (str): The prompt template with the scoring instructions.
            metrics (list[Metric | dict[str, list[Metric]]] | dict[str, list[Metric]] | None):
                The metrics to use for the evaluation. If `None`, the default metric
                will be used (G-Eval with mean aggregation).
            logprobs (bool): Use a one-token conditional mean over reported valid
                rating alternatives. Every allowed rating must be present; missing
                or inconsistent evidence raises ValueError. False uses the sampled
                integer without a one-token limit.
            top_logprobs (int): Positive integer alternative count, at least the
                number of allowed ratings in weighted mode. Providers may impose
                their own limit or omit low-probability ratings.
            min_score (int): The minimum valid score.
            max_score (int): The maximum valid score.
            normalise (bool): Whether to normalise the scores between 0 and 1.
            debug (bool): Retained for compatibility. Weighting failures always raise.
        """
        validate_score_range(min_score, max_score, normalise=normalise)
        validate_weighting_settings(logprobs, top_logprobs, min_score, max_score)
        self.name = name
        self.prompt_template = prompt_template
        if metrics is None:
            metrics = [mean()]
        self.metrics = metrics
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.min_score = min_score
        self.max_score = max_score
        self.normalise = normalise
        self.debug = debug

    @override
    def create_scorer(self, model: Model) -> Scorer:
        """
        Creates a G-Eval scorer.

        Args:
            model (Model): The model to create a scorer for.

        Returns:
            Scorer: The created G-Eval scorer.
        """

        @scorer(name=self.name, metrics=self.metrics)
        def g_eval_scorer() -> Scorer:
            g_eval_calculator = GEvalScoreCalculator(
                model=model,
                prompt_template=self.prompt_template,
                logprobs=self.logprobs,
                top_logprobs=self.top_logprobs,
                min_score=self.min_score,
                max_score=self.max_score,
                normalise=self.normalise,
                debug=self.debug,
            )

            async def score(state: TaskState, target: Target):
                return await g_eval_calculator.calculate_async(
                    input=state.input_text,
                    prediction=state.output.completion,
                    reference=target.text,
                    metadata=state.metadata,
                )

            return score

        return g_eval_scorer()


def get_g_eval_evaluator(
    *,
    name: str = "G-Eval",
    quality_name: str = "Unknown",
    model_name: str | None = None,
    metrics: list[Metric | dict[str, list[Metric]]]
    | dict[str, list[Metric]]
    | None = None,
    prompt_template: str,
    model_config: ModelConfig,
    logprobs: bool = True,
    top_logprobs: int = 20,
    min_score: int = 1,
    max_score: int = 10,
    normalise: bool = True,
    debug: bool = False,
    version: str | None = None,
) -> Evaluator:
    """
    Constructs a G-Eval evaluator that can be used in EvalSense evaluation pipeline.

    Args:
        name (str): The name of the evaluator. Defaults to "G-Eval".
        quality_name (str): The name of the quality to be evaluated by G-Eval.
        model_name (str | None): The name of the model to be used for evaluation.
            If `None`, the model name will be taken from the model config.
        metrics (list[Metric | dict[str, list[Metric]]] | dict[str, list[Metric]] | None):
            The metrics to use for the evaluation. If `None`, the default metric
            will be used (G-Eval).
        prompt_template (str): The prompt template to use. The supplied template should
            be a format string with {prediction} and (optionally) {reference} as
            placeholders, as well as any additional placeholders for entries in
            Inspect AI sample/task state metadata. The template should instruct the
            judge model to return only one non-negative integer between min_score
            and max_score. Fractions, decimals, signs and explanatory text fail
            validation. Outer whitespace is accepted. Leading zeroes are accepted
            only when logprobs=False; weighted mode requires canonical labels.
        model_config (ModelConfig): The model configuration.
        logprobs (bool): Use a one-token conditional mean over reported valid
            rating alternatives, requiring evidence for every allowed rating.
            Missing evidence raises ValueError without a sampled-score fallback.
            False explicitly uses the sampled integer.
        top_logprobs (int): Positive integer alternative count, at least the
            number of allowed ratings in weighted mode. This budget does not
            guarantee the provider returns every rating.
        min_score (int): The minimum valid score.
        max_score (int): The maximum valid score.
        normalise (bool): Whether to normalise the scores between 0 and 1.
        debug (bool): Retained for compatibility. Weighting failures always raise.

        version (str | None): Cache version for custom code or external resources.

    Returns:
        Evaluator: The constructed G-Eval evaluator. Cache identity includes the
            estimator format and, for weighted mode, the weighting policy.
    """
    cache_parameters = {
        "method": "g_eval",
        "score_format": SCORE_FORMAT,
        "estimator_format": ESTIMATOR_FORMAT,
        "quality_name": quality_name,
        "metrics": metrics,
        "prompt_template": prompt_template,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "min_score": min_score,
        "max_score": max_score,
        "normalise": normalise,
        "debug": debug,
    }
    if logprobs:
        cache_parameters["weighting_policy"] = WEIGHTING_POLICY
    metric_name = f"{name} ({quality_name}, {model_name or model_config.name})"
    return Evaluator(
        version=version,
        cache_parameters=cache_parameters,
        name=metric_name,
        scorer=GEvalScorerFactory(
            name=metric_name,
            metrics=metrics,
            prompt_template=prompt_template,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            min_score=min_score,
            max_score=max_score,
            normalise=normalise,
            debug=debug,
        ),
        model_config=model_config,
    )
