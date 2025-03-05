from typing import Protocol

from datasets import Dataset, DatasetDict


class TaskPreprocessor(Protocol):
    """A protocol for preprocessing datasets for a specific task."""

    def __call__(
        self, dataset: Dataset | DatasetDict, **kwargs
    ) -> Dataset | DatasetDict:
        """Preprocesses the input dataset for a specific task.

        Args:
            dataset (Dataset | DatasetDict): The input dataset to preprocess.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            (Dataset | DatasetDict): The preprocessed dataset.
        """
        ...
