from typing import Protocol, runtime_checkable

from datasets import Dataset, DatasetDict


@runtime_checkable
class TaskPreprocessor(Protocol):
    """A protocol for preprocessing datasets for a specific task."""

    def __call__(
        self, dataset: Dataset | DatasetDict, **kwargs
    ) -> Dataset | DatasetDict:
        """Preprocesses the input dataset for a specific task.

        Args:
            dataset (Dataset | DatasetDict): The input dataset to preprocess.
            **kwargs: Additional keyword arguments.

        Returns:
            (Dataset | DatasetDict): The preprocessed dataset.
        """
        ...
