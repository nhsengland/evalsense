from pathlib import Path
from typing import Protocol

import datasets
from inspect_ai.dataset import Dataset, FieldSpec, RecordToSample, json_dataset

from evalsense.datasets.dataset_manager import DatasetManager
from evalsense.utils.huggingface import disable_dataset_progress_bars
from evalsense.utils.files import to_safe_filename
from evalsense.utils.cache import CacheIdentityError, fingerprint


def _dataset_fingerprint(hf_dataset: datasets.Dataset) -> str:
    """Read the upstream dataset-state fingerprint through one checked boundary."""
    # Hugging Face documents _fingerprint as its cache identity. Keep access here
    # so an upstream API change cannot fall back to a name-only identity.
    value = getattr(hf_dataset, "_fingerprint", None)
    if not isinstance(value, str) or not value.strip():
        raise CacheIdentityError(
            "Dataset fingerprint is missing or invalid; cannot identify "
            "preprocessed data safely. Reload the dataset using a supported "
            "datasets version."
        )
    return value


class TaskPreprocessingFunction(Protocol):
    """A protocol for a function that preprocesses datasets.

    You can pass this function to a TaskPreprocessor to perform some
    task-specific preprocessing on a dataset. This is especially useful
    in cases in which a single dataset can be used for multiple different
    tasks, with each requiring different preprocessing steps.
    """

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
        *,
        version: str | None = None,
    ) -> None:
        """Initializes the task preprocessor.

        Args:
            name (str): The name of the task preprocessor.
            preprocessing_function (TaskPreprocessingFunction): The function used to
                preprocess the dataset.
            version (str | None): Required for custom code. Bump when its behavior
                or external inputs change.
        """
        self.name = name
        self.preprocessing_function = preprocessing_function
        self.version = version

    @property
    def cache_key(self) -> str:
        """Version custom preprocessing rather than guessing its implementation."""
        return fingerprint(
            {"implementation": type(self), "function": self.preprocessing_function},
            version=self.version,
        )

    def __call__(
        self,
        hf_dataset: datasets.Dataset,
        dataset_manager: DatasetManager,
        field_spec: FieldSpec | RecordToSample | None = None,
        force_reprocess: bool = False,
    ) -> Dataset:
        """Preprocesses the input dataset for a specific task.

        Args:
            hf_dataset (datasets.Dataset): The input dataset to preprocess,
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
        data_key = fingerprint(
            {
                "dataset": dataset_manager.record,
                "content": _dataset_fingerprint(hf_dataset),
                "preprocessor": self.cache_key,
            }
        )
        dataset_filename = f"{to_safe_filename(self.name)[:40]}-{data_key}.jsonl"
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

    def __init__(self, name="Identity", *, version: str | None = None) -> None:
        """Initializes the default task preprocessor."""
        super().__init__(
            name=name,
            preprocessing_function=default_task_preprocessing_function,
            version=version,
        )
