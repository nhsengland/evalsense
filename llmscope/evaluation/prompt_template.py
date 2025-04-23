from abc import abstractmethod
from typing import Protocol


class EvalPromptTemplate(Protocol):
    """A protocol for constructing prompts for evaluation."""

    @abstractmethod
    def __call__(self, prediction: str, reference: str | None, **kwargs: dict) -> str:
        """Constructs a prompt string for the given task state and target."""
        pass
