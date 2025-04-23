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
from llmscope.evaluation import Evaluator, EvalPromptTemplate, ScorerFactory
from llmscope.generation import ModelConfig
from llmscope.logging import get_logger
from llmscope.utils.evaluation import extract_score, extract_weighted_score

logger = get_logger(__name__)


def g_eval_base_factory(
    *,
    prompt_template: EvalPromptTemplate,
    logprobs: bool = True,
    top_logprobs: int = 20,
    min_score: int = 1,
    max_score: int = 10,
    normalise: bool = True,
    model: Model,
) -> Callable[[], Scorer]:
    """
    Base factory function to create a G-Eval scorer.

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
        Callable[[], Scorer]: The G-Eval scorer factory function.
    """
    logprobs_config = GenerateConfig(
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    )
    config = model.config.merge(logprobs_config)

    def g_eval_base():
        async def score(state: TaskState, target: Target):
            llm_input = prompt_template(state, target)
            output = await model.generate(llm_input, config=config)

            score = extract_score(output.completion, min_score, max_score)
            if logprobs:
                try:
                    score = extract_weighted_score(
                        output, score, min_score=min_score, max_score=max_score
                    )
                except ValueError as e:
                    logger.error(
                        f"Cannot compute weighted evaluation score: {e}. "
                        "Falling back to standard score."
                    )

            if normalise:
                score = (score - min_score) / (max_score - min_score)

            return Score(
                value=score,
                answer=state.output.completion,
            )

        return score

    return g_eval_base


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
    Factory function to create a full G-Eval scorer.

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
    g_eval_base = g_eval_base_factory(
        prompt_template=prompt_template,
        model=model,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        min_score=min_score,
        max_score=max_score,
        normalise=normalise,
    )

    @scorer(name=f"G-Eval ({name}, {model.name})", metrics=[mean()])
    def g_eval() -> Scorer:
        return g_eval_base()

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
