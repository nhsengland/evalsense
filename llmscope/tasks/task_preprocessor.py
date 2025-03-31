from pathlib import Path
from typing import Protocol

import datasets
from inspect_ai.dataset import Dataset, FieldSpec, RecordToSample, json_dataset

from llmscope.datasets.dataset_manager import DatasetManager
from llmscope.utils.huggingface import disable_dataset_progress_bars
from llmscope.utils.files import to_safe_filename


class TaskPreprocessingFunction(Protocol):
    """A protocol for a function that preprocesses datasets."""

    def __call__(
        self, hf_dataset: datasets.Dataset, dataset_manager: DatasetManager
    ) -> datasets.Dataset:
        """Preprocesses the input dataset for a specific task.

        Args:
            hf_dataset (datasets.Dataset): The input dataset to preprocess,
                in HuggingFace format.
            dataset_manager (DatasetManager): The dataset manager used to
                retrieve the dataset.

        Returns:
            (datasets.Dataset): The preprocessed dataset.
        """
        ...


class TaskPreprocessor:
    """A class preprocessing a dataset for a specific task."""

    def __init__(
        self,
        name: str,
        preprocessing_function: TaskPreprocessingFunction,
    ) -> None:
        """Initializes the task preprocessor.

        Args:
            name (str): The name of the task preprocessor.
            preprocessing_function (TaskPreprocessingFunction): The function used to
                preprocess the dataset.
        """
        self.name = name
        self.preprocessing_function = preprocessing_function

    def __call__(
        self,
        hf_dataset: datasets.Dataset,
        dataset_manager: DatasetManager,
        field_spec: FieldSpec | RecordToSample | None = None,
        force_reprocess: bool = False,
    ) -> Dataset:
        """Preprocesses the input dataset for a specific task.

        Args:
            dataset (datasets.Dataset): The input dataset to preprocess,
                in HuggingFace format.
            dataset_manager (DatasetManager): The dataset manager used to
                retrieve the dataset.
            field_spec (FieldSpec): Specification mapping dataset fields to
                sample fields. See Inspect AI documentation for more details.
            force_reprocess (bool): Whether to force reprocess the dataset
                even if it already exists. Defaults to False.

        Returns:
            (Dataset): The preprocessed dataset.
        """
        tasks_path = Path(dataset_manager.version_path / "tasks")
        tasks_path.mkdir(parents=True, exist_ok=True)
        dataset_filename = (
            f"{to_safe_filename(dataset_manager.name)}-"
            f"{to_safe_filename(self.name)}.jsonl"
        )
        task_data_path = tasks_path / dataset_filename

        if not task_data_path.exists() or force_reprocess:
            preprocessed_hf_dataset = self.preprocessing_function(
                hf_dataset, dataset_manager
            )
            with disable_dataset_progress_bars():
                preprocessed_hf_dataset.to_json(task_data_path, lines=True)

        dataset = json_dataset(str(task_data_path), sample_fields=field_spec)
        return dataset


def default_task_preprocessing_function(
    hf_dataset: datasets.Dataset, dataset_manager: DatasetManager
) -> datasets.Dataset:
    """Default preprocessing function that returns the input dataset unchanged."""
    return hf_dataset


class DefaultTaskPreprocessor(TaskPreprocessor):
    """Default task preprocessor that returns the input dataset unchanged."""

    def __init__(self, name="Identity") -> None:
        """Initializes the default task preprocessor."""
        super().__init__(
            name=name,
            preprocessing_function=default_task_preprocessing_function,
        )
