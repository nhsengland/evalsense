from abc import abstractmethod
from typing import Protocol

from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState


class EvalPromptTemplate(Protocol):
    """A protocol for constructing prompts for evaluation."""

    @abstractmethod
    def __call__(self, state: TaskState, target: Target) -> str:
        """Constructs a prompt string for the given task state and target."""
        pass
