from typing import Protocol

from datasets import Dataset, DatasetDict


class TaskPreprocessingFunction[T: (Dataset, DatasetDict)](Protocol):
    """A protocol for a function that preprocesses datasets."""

    def __call__(self, dataset: T) -> T:
        """Preprocesses the input dataset for a specific task.

        Args:
            dataset (Dataset | DatasetDict): The input dataset to preprocess.

        Returns:
            (Dataset | DatasetDict): The preprocessed dataset.
        """
        ...


class TaskPreprocessor[T: (Dataset, DatasetDict)]:
    """A protocol for preprocessing datasets for a specific task."""

    def __init__(
        self, name: str, preprocessing_function: TaskPreprocessingFunction[T]
    ) -> None:
        """Initializes the task preprocessor.

        Args:
            name (str): The name of the task preprocessor.
            preprocessing_function (TaskPreprocessingFunction): The function used to
                preprocess the dataset.
        """
        self.name = name
        self.preprocessing_function = preprocessing_function

    def __call__(self, dataset: T) -> T:
        """Preprocesses the input dataset for a specific task.

        Args:
            dataset (Dataset | DatasetDict): The input dataset to preprocess.

        Returns:
            (Dataset | DatasetDict): The preprocessed dataset.
        """
        return self.preprocessing_function(dataset)
