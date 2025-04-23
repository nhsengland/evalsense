from typing import Callable, override

from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    mean,
    scorer,
)
from inspect_ai.solver import TaskState

from llmscope.evaluation import (
    Evaluator,
    EvalPromptTemplate,
    ScoreCalculator,
    ScorerFactory,
)
from llmscope.generation import ModelConfig
from llmscope.logging import get_logger
from llmscope.utils.evaluation import extract_score, extract_weighted_score

logger = get_logger(__name__)


class GEvalScoreCalculator(ScoreCalculator):
    def __init__(
        self,
        model: Model,
        prompt_template: EvalPromptTemplate,
        logprobs: bool = True,
        top_logprobs: int = 20,
        min_score: int = 1,
        max_score: int = 10,
        normalise: bool = True,
    ):
        self.model = model
        self.prompt_template = prompt_template
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.min_score = min_score
        self.max_score = max_score
        self.normalise = normalise

    @override
    def calculate(
        self, *, prediction: str, reference: str | None = None, **kwargs: dict
    ) -> Score:
        raise NotImplementedError(
            "Synchronous evaluation is not supported for G-Eval. "
            "Use calculate_async instead."
        )

    @override
    async def calculate_async(
        self, *, prediction: str, reference: str | None = None, **kwargs: dict
    ) -> Score:
        logprobs_config = GenerateConfig(
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
        )
        llm_input = self.prompt_template(prediction=prediction, reference=reference)
        output = await self.model.generate(llm_input, config=logprobs_config)

        score = extract_score(output.completion, self.min_score, self.max_score)
        if self.logprobs:
            try:
                score = extract_weighted_score(
                    output, score, min_score=self.min_score, max_score=self.max_score
                )
            except ValueError as e:
                logger.error(
                    f"Cannot compute weighted evaluation score: {e}. "
                    "Falling back to standard score."
                )

        if self.normalise:
            score = (score - self.min_score) / (self.max_score - self.min_score)

        return Score(
            value=score,
            answer=prediction,
        )


def g_eval_base(
    *,
    prompt_template: EvalPromptTemplate,
    logprobs: bool = True,
    top_logprobs: int = 20,
    min_score: int = 1,
    max_score: int = 10,
    normalise: bool = True,
    model: Model,
) -> Scorer:
    """
    Base scorer for G-Eval.

    Args:
        prompt_template (EvalPromptTemplate): The prompt template to use.
        logprobs (bool): Whether to use model log probabilities to compute weighted
            evaluation score instead of a standard score.
        top_logprobs (int): The number of top log probabilities to consider.
        min_score (int): The minimum valid score.
        max_score (int): The maximum valid score.
        normalise (bool): Whether to normalise the scores between 0 and 1.
        model (Model): The model to use for evaluation.

    Returns:
        Scorer: The G-Eval scorer factory function.
    """

    async def score(state: TaskState, target: Target):
        g_eval_calculator = GEvalScoreCalculator(
            model=model,
            prompt_template=prompt_template,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            min_score=min_score,
            max_score=max_score,
            normalise=normalise,
        )
        return await g_eval_calculator.calculate_async(
            prediction=state.output.completion, reference=target.text
        )

    return score


def g_eval_factory(
    *,
    name: str,
    prompt_template: EvalPromptTemplate,
    logprobs: bool = True,
    top_logprobs: int = 20,
    min_score: int = 1,
    max_score: int = 10,
    normalise: bool = True,
    model: Model,
) -> Callable[[], Scorer]:
    """
    Factory function to create a G-Eval scorer.

    Args:
        name (str): The name of the scorer.
        prompt_template (EvalPromptTemplate): The prompt template to use.
        logprobs (bool): Whether to use model log probabilities to compute weighted
            evaluation score instead of a standard score.
        top_logprobs (int): The number of top log probabilities to consider.
        min_score (int): The minimum valid score.
        max_score (int): The maximum valid score.
        normalise (bool): Whether to normalise the scores between 0 and 1.
        model (Model): The model to use for evaluation.

    Returns:
        Callable[[], Scorer]: The G-Eval scorer factory function.
    """

    @scorer(name=f"G-Eval ({name}, {model.name})", metrics=[mean()])
    def g_eval() -> Scorer:
        return g_eval_base(
            prompt_template=prompt_template,
            model=model,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            min_score=min_score,
            max_score=max_score,
            normalise=normalise,
        )

    return g_eval


class GEvalScorerFactory(ScorerFactory):
    """Scorer factory for G-Eval."""

    def __init__(
        self,
        name: str,
        prompt_template: EvalPromptTemplate,
        logprobs: bool = True,
        top_logprobs: int = 20,
        min_score: int = 1,
        max_score: int = 10,
        normalise: bool = True,
    ):
        """
        Initialize the G-Eval scorer factory.

        Args:
            name (str): The name of the scorer.
            prompt_template (EvalPromptTemplate): The prompt template to use.
            logprobs (bool): Whether to use model log probabilities to compute weighted
                evaluation score instead of a standard score.
            top_logprobs (int): The number of top log probabilities to consider.
            min_score (int): The minimum valid score.
            max_score (int): The maximum valid score.
            normalise (bool): Whether to normalise the scores between 0 and 1.
        """
        self._create_scorer = lambda model: g_eval_factory(
            name=name,
            prompt_template=prompt_template,
            model=model,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            min_score=min_score,
            max_score=max_score,
            normalise=normalise,
        )()

    @override
    def create_scorer(self, model: Model) -> Scorer:
        return self._create_scorer(model)


def get_g_eval_evaluator(
    *,
    name: str = "G-Eval",
    prompt_template: EvalPromptTemplate,
    model_config: ModelConfig,
    logprobs: bool = True,
    top_logprobs: int = 20,
    min_score: int = 1,
    max_score: int = 10,
    normalise: bool = True,
) -> Evaluator:
    """
    Constructs a G-Eval evaluator that can be used in LLMScope evaluation pipeline.

    Args:
        name (str): The name of the evaluator.
        prompt_template (EvalPromptTemplate): The prompt template to use.
        model_config (ModelConfig): The model configuration.
        logprobs (bool): Whether to use model log probabilities to compute weighted
            evaluation score instead of a standard score.
        top_logprobs (int): The number of top log probabilities to consider.
        min_score (int): The minimum valid score.
        max_score (int): The maximum valid score.
        normalise (bool): Whether to normalise the scores between 0 and 1.

    Returns:
        Evaluator: The constructed G-Eval evaluator.
    """
    return Evaluator(
        name=name,
        scorer=GEvalScorerFactory(
            name=name,
            prompt_template=prompt_template,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            min_score=min_score,
            max_score=max_score,
            normalise=normalise,
        ),
        model_config=model_config,
    )
