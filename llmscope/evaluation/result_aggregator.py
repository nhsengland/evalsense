from abc import abstractmethod
from enum import Enum
from typing import Protocol

from llmscope.evaluation.evaluation_result import EvaluationResult
from llmscope.evaluation.experiment import ExperimentId


class ResultCategory(Enum):
    """Built-in result categories."""

    STATISTICAL = "statistical"
    ALIGNMENT = "alignment"
    FACTUALITY = "factuality"
    CONCISENESS = "conciseness"
    OTHER = "other"


class ResultAggregator[T](Protocol):
    """A protocol for aggregating evaluation results."""

    @abstractmethod
    def add_result(
        self,
        *,
        result: EvaluationResult,
        experiment_id: ExperimentId,
        exist_ok: bool = False,
    ) -> None:
        """Adds a result to the aggregator.

        Args:
            result (EvaluationResult): The evaluation result to add.
            experiment_id (ExperimentId): The ID data of the experiment associated
                with the result.
            exist_ok (bool, optional): Whether to allow adding the same result
                multiple times. Defaults to False.
        """
        ...

    @abstractmethod
    def return_results(self) -> T:
        """Aggregates the results.

        Returns:
            T: The aggregated result.
        """
        ...
