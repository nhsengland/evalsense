from abc import abstractmethod
from enum import Enum
from typing import Protocol, TypeVar

from datasets import Dataset

from llmscope.evaluation.evaluation_result import EvaluationResult


class ResultCategory(Enum):
    """Built-in result categories."""

    STATISTICAL = "statistical"
    ALIGNMENT = "alignment"
    FACTUALITY = "factuality"
    CONCISENESS = "conciseness"
    OTHER = "other"


T = TypeVar("T", covariant=True)


class ResultAggregator(Protocol[T]):
    """A protocol for aggregating evaluation results."""

    @abstractmethod
    def add_result(
        self,
        *,
        result: EvaluationResult,
        dataset: Dataset,
        dataset_name: str,
        task_name: str,
        prompt_name: str,
        model_name: str,
    ) -> None:
        """Adds a result to the aggregator.

        Args:
            result (EvaluationResult): The evaluation result to add.
            dataset (Dataset): The dataset associated with the result.
            dataset_name (str): The name of the dataset.
            task_name (str): The name of the task.
            prompt_name (str): The name of the prompt.
            model_name (str): The name of the model.
        """
        ...

    @abstractmethod
    def return_results(self) -> T:
        """Aggregates the results.

        Returns:
            T: The aggregated result.
        """
        ...
