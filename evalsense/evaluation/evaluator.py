from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from inspect_ai.model import Model
from inspect_ai.scorer import Score, Scorer

from evalsense.generation import ModelConfig


@runtime_checkable
class ScoreCalculator(Protocol):
    """A protocol for computing evaluation scores."""

    @abstractmethod
    def calculate(
        self,
        *,
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: dict,
    ) -> Score:
        """Computes evaluation scores for the given evaluation method

        Args:
            prediction (str): The model output to evaluate.
            input (str, optional): The input to the model. Optional.
            reference (str, optional): The reference output to compare against.
                Optional.
            metadata (dict[str, Any], optional): Additional Inspect AI sample/task
                state metadata. Optional.
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
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: dict,
    ) -> Score:
        """Asynchronously computes evaluation scores for the given evaluation method

        Args:
            prediction (str): The model output to evaluate.
            input (str, optional): The input to the model. Optional.
            reference (str, optional): The reference output to compare against.
                Optional.
            metadata (dict[str, Any], optional): Additional Inspect AI sample/task
                state metadata. Optional.
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
    cleanup_fun: Callable[[], None] | None = None

    @property
    def model_name(self) -> str:
        """Retrieves the model name associated with the evaluator config.

        Returns an empty string if the evaluator doesn't use a model config.

        Returns:
            str: The name of the model in the config or empty string.
        """
        if self.model_config is None:
            return ""
        return self.model_config.name
