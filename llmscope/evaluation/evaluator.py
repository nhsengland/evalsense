from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from inspect_ai.model import Model
from inspect_ai.scorer import Score, Scorer

from llmscope.generation import ModelConfig


@runtime_checkable
class ScoreCalculator(Protocol):
    """A protocol for computing evaluation scores."""

    @abstractmethod
    def calculate(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        **kwargs: dict,
    ) -> Score:
        """Computes evaluation scores for the given evaluation method

        Args:
            predictions (str): The model prediction to evaluate.
            references (str, optional): The reference output to compare against.
            **kwargs (dict): Additional keyword arguments specific to the given
                evaluation method.

        Returns:
            Score: The Inspect AI Score object with the calculated result.
        """
        pass

    @abstractmethod
    async def calculate_async(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        **kwargs: dict,
    ) -> Score:
        """Computes evaluation scores for the given evaluation method

        Args:
            predictions (str): The model prediction to evaluate.
            references (str, optional): The reference output to compare against.
            **kwargs (dict): Additional keyword arguments specific to the given
                evaluation method.

        Returns:
            Score: The Inspect AI Score object with the calculated result.
        """
        pass


@runtime_checkable
class ScorerFactory(Protocol):
    """A protocol for constructing a Scorer given a Model."""

    @abstractmethod
    def create_scorer(self, model: Model) -> Scorer:
        """Creates a Scorer from a Model.

        Args:
            model (Model): The model to create a scorer for.

        Returns:
            Scorer: The created scorer.
        """
        pass


@dataclass
class Evaluator:
    """A class for LLM output evaluators."""

    name: str
    scorer: Scorer | ScorerFactory
    model_config: ModelConfig | None = None
