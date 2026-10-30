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
from evalsense.evaluation.judge_prompt import (
    JudgeInputError,
    JudgeInputPolicy,
    JudgePrompt,
    PROTOCOL,
    validate_judge_data,
    generate_judge,
)
from evalsense.generation import ModelConfig
from evalsense.logging import get_logger
from evalsense.utils.text import extract_score, extract_weighted_score

logger = get_logger(__name__)


class GEvalScoreCalculator(ScoreCalculator):
    """G-Eval with trusted template instructions and JSON-encoded sample fields.

    Templates and provider configuration are trusted evaluator code. Sample data
    never fills the system instructions. Complete in-range integer responses are
    required before the existing optional log-probability weighting is applied.
    Isolation and optional instruction screening are mitigations, not proof of
    injection resistance. Re-score under a new evaluator identity when migrating.
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
        *,
        input_policy: JudgeInputPolicy = "isolate",
    ):
        """
        Initializes the G-Eval score calculator.

        Args:
            model (Model): The model to use for evaluation.
            prompt_template (str): The prompt template with the scoring instructions.
            logprobs (bool): Whether to use model log probabilities to compute weighted
                evaluation score instead of a standard score.
            top_logprobs (int): The number of top log probabilities to consider.
            min_score (int): The minimum valid score.
            max_score (int): The maximum valid score.
            normalise (bool): Whether to normalise the scores between 0 and 1.
            debug (bool): Whether to report repeated errors in the log.
            input_policy (JudgeInputPolicy): Isolate evidence, or also reject known
                instruction phrases. Screening is heuristic and may flag quotations.
        """
        self.model = model
        self.prompt_template = prompt_template
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.min_score = min_score
        self.max_score = max_score
        self.normalise = normalise
        self.debug = debug
        validate_judge_data(None, input_policy)
        self.input_policy: JudgeInputPolicy = input_policy
        self.warned_weighted_score = False

    @override
    def calculate(
        self,
        *,
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
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
        **kwargs: Any,
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
        """
        logprobs_config = GenerateConfig(
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
        )
        validate_judge_data(
            {
                "prediction": prediction,
                "reference": reference,
                "input": input,
                "metadata": metadata,
            },
            self.input_policy,
        )
        if metadata and {"prediction", "reference", "input"}.intersection(metadata):
            raise JudgeInputError(
                "Sample metadata must not override reserved judge fields."
            )
        prompt = JudgePrompt.from_template(
            self.prompt_template,
            prediction=prediction,
            reference=reference,
            input=input,
            **(metadata or {}),
        )
        output, llm_input = await generate_judge(
            self.model,
            prompt,
            logprobs_config,
            policy=self.input_policy,
            kind="rating",
            min_score=self.min_score,
            max_score=self.max_score,
        )

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
                    error_message += f" Offending output: {output.completion}"

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
                "judge_input_protocol": PROTOCOL,
                "judge_input_policy": self.input_policy,
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
        *,
        input_policy: JudgeInputPolicy = "isolate",
    ):
        """
        Initialize the G-Eval scorer factory.

        Args:
            name (str): The name of the scorer.
            prompt_template (str): The prompt template with the scoring instructions.
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
            input_policy (JudgeInputPolicy): Isolate evidence, or also reject known
                instruction phrases. Screening is heuristic and may flag quotations.
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
        validate_judge_data(None, input_policy)
        self.input_policy: JudgeInputPolicy = input_policy

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
                input_policy=self.input_policy,
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
    input_policy: JudgeInputPolicy = "isolate",
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
        input_policy (JudgeInputPolicy): Isolate evidence, or also reject known
            instruction phrases. Does not guarantee injection resistance.

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
            input_policy=input_policy,
        ),
        model_config=model_config,
    )
