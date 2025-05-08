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
from evalsense.logging import get_logger
from evalsense.utils.text import extract_score, extract_weighted_score, format_template

logger = get_logger(__name__)


class GEvalScoreCalculator(ScoreCalculator):
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
        self.model = model
        self.prompt_template = prompt_template
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.min_score = min_score
        self.max_score = max_score
        self.normalise = normalise
        self.debug = debug
        self.warned_weighted_score = False

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
        **kwargs: dict,
    ) -> Score:
        logprobs_config = GenerateConfig(
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
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
        if self.logprobs:
            try:
                raw_score = extract_weighted_score(
                    output, min_score=self.min_score, max_score=self.max_score
                )
            except ValueError as e:
                if not self.warned_weighted_score or self.debug:
                    self.warned_weighted_score = True

                    error_message = (
                        f"❌  Cannot compute weighted evaluation score: {e} "
                        "Falling back to standard score."
                    )

                    if not self.debug:
                        error_message += (
                            " Further errors will be suppressed "
                            + "(set debug=True to see all errors)."
                        )

                    logger.error(error_message)

        if self.normalise:
            score = (raw_score - self.min_score) / (self.max_score - self.min_score)
        else:
            score = raw_score

        return Score(
            value=score,
            answer=prediction,
            metadata={
                "prompt": llm_input,
                "output_text": output.completion,
                "raw_score": raw_score,
            },
        )


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
            prompt_template (str): The prompt template to use.
            metrics (list[Metric | dict[str, list[Metric]]] | dict[str, list[Metric]] | None):
                The metrics to use for the evaluation. If `None`, the default metric
                will be used (G-Eval with mean aggregation).
            logprobs (bool): Whether to use model log probabilities to compute weighted
                evaluation score instead of a standard score.
            top_logprobs (int): The number of top log probabilities to consider.
            min_score (int): The minimum valid score.
            max_score (int): The maximum valid score.
            normalise (bool): Whether to normalise the scores between 0 and 1.
            debug (bool): Whether to report repeated errors in the log.
        """
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
            judge model to respond with a numerical score between the specified
            min_score and max_score.
        model_config (ModelConfig): The model configuration.
        logprobs (bool): Whether to use model log probabilities to compute weighted
            evaluation score instead of a standard score.
        top_logprobs (int): The number of top log probabilities to consider.
        min_score (int): The minimum valid score.
        max_score (int): The maximum valid score.
        normalise (bool): Whether to normalise the scores between 0 and 1.
        debug (bool): Whether to report repeated errors in the log.

    Returns:
        Evaluator: The constructed G-Eval evaluator.
    """
    metric_name = f"{name} ({quality_name}, {model_name or model_config.name})"
    return Evaluator(
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
