from typing import Protocol, TypeVar

from datasets import Dataset, DatasetDict

T = TypeVar("T", Dataset, DatasetDict)


class TaskPreprocessor(Protocol[T]):
    """A protocol for preprocessing datasets for a specific task."""

    def __call__(self, dataset: T) -> T:
        """Preprocesses the input dataset for a specific task.

        Args:
            dataset (Dataset | DatasetDict): The input dataset to preprocess.

        Returns:
            (Dataset | DatasetDict): The preprocessed dataset.
        """
        ...
