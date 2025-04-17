from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from inspect_ai.model import Model
from inspect_ai.scorer import Scorer

from llmscope.generation import ModelConfig


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
