from abc import ABC, abstractmethod
from pathlib import Path
import shutil

from datasets import Dataset, DatasetDict, load_from_disk

from llmscope.constants import DEFAULT_VERSION_NAME, DATA_PATH
from llmscope.datasets.dataset_config import DatasetConfig
from llmscope.utils.files import to_safe_filename, download_file


# TODO: Consider making the config optional by splitting out ConfigDatasetManager
class DatasetManager(ABC):
    """An abstract class for managing datasets.

    Attributes:
        name (str): The name of the dataset.
        config (DatasetConfig): The configuration for the dataset.
        version (str): The used dataset version.
        priority (int): The priority of the dataset manager.
        data_path (Path): The top-level directory for storing all datasets.
    """

    def __init__(
        self,
        name: str,
        version: str = DEFAULT_VERSION_NAME,
        priority: int = 10,
        data_dir: str | None = None,
        **kwargs,
    ):
        """Initializes a new DatasetManager.

        Args:
            name (str): The name of the dataset.
            version (str): The dataset version to retrieve.
            priority (int, optional): The priority of the dataset manager when
                choosing between multiple possible managers. Recommended values
                range from 0 to 10, with 10 (the highest) being the default.
            data_dir (str, optional): The top-level directory for storing all
                datasets. Defaults to "datasets" in the user cache directory.
            **kwargs (Any): Additional keyword arguments.
        """
        self.name = name
        self.config = DatasetConfig(name)
        self.version = version
        self.priority = priority
        if data_dir is not None:
            self.data_path = Path(data_dir)
        else:
            self.data_path = DATA_PATH

    @property
    def dataset_path(self) -> Path:
        """The top-level directory for storing this dataset.

        Returns:
            (Path): The dataset directory.
        """
        return self.data_path / to_safe_filename(self.name)

    @property
    def version_path(self) -> Path:
        """The directory for storing a specific version of this dataset.

        Returns:
            (Path): The dataset version directory.
        """
        return self.dataset_path / to_safe_filename(self.version)

    @property
    def main_data_path(self) -> Path:
        """The path for storing the main dataset files for a specific version.

        Returns:
            (Path): The main dataset directory.
        """
        return self.version_path / "main"

    def _retrieve_files(self, splits: list[str], **kwargs) -> None:
        """Retrieves  dataset files.

        This method retrieves all the dataset files for the specified splits
        into the `self.version_path` directory.

        Args:
            splits (list[str]): The dataset splits to retrieve.
            **kwargs (Any): Additional keyword arguments.
        """
        for filename, file_metadata in self.config.get_files(
            self.version, splits
        ).items():
            effective_source = file_metadata.effective_source
            if effective_source is not None and effective_source.online:
                download_file(
                    effective_source.url_template.format(
                        version=self.version, filename=filename
                    ),
                    self.version_path / filename,
                    expected_hash=file_metadata.hash,
                    hash_type=file_metadata.hash_type,
                )

    @abstractmethod
    def _preprocess_files(self, splits: list[str], **kwargs) -> None:
        """Preprocesses the downloaded dataset files.

        This method preprocesses the retrieved dataset files and saves them
        as a HuggingFace DatasetDict in the `self.main_data_path` directory.

        Args:
            splits (list[str]): The dataset splits to preprocess.
            **kwargs (Any): Additional keyword arguments.
        """
        pass

    def get(self, splits: list[str] | None = None, **kwargs) -> None:
        """Downloads and preprocesses a dataset.

        Args:
            splits (list[str], optional): The dataset splits to retrieve.
            **kwargs (Any): Additional keyword arguments.
        """
        if splits is None:
            splits = self.config.get_splits(self.version).keys()

        self.version_path.mkdir(parents=True, exist_ok=True)
        self._retrieve_files(splits=splits, **kwargs)
        self._preprocess_files(splits=splits, **kwargs)

    def is_retrieved(self) -> bool:
        """Checks if the dataset at the specific version is already downloaded.

        Returns:
            (bool): True if the dataset exists locally, False otherwise.
        """
        return self.main_data_path.exists()

    def remove(self) -> None:
        """Deletes the dataset at the specific version from disk."""
        if self.version_path.exists():
            shutil.rmtree(self.version_path)

    def load(
        self, splits: list[str] | None = None, retrieve=True
    ) -> DatasetDict | Dataset:
        """Loads the dataset as a HuggingFace dataset.

        Args:
            splits (list[str], optional): The dataset splits to load.
            retrieve (bool, optional): Whether to retrieve the dataset if it
                does not exist locally. Defaults to True.

        Returns:
            (DatasetDict): The loaded dataset.
        """
        if not self.is_retrieved() and retrieve:
            self.get(splits=splits)
        hf_dataset = load_from_disk(self.main_data_path)
        if splits is not None:
            if len(splits) == 1:
                hf_dataset = hf_dataset[splits[0]]
            else:
                hf_dataset = hf_dataset[splits]
        return hf_dataset

    def __call__(
        self, splits: list[str] | None = None, retrieve=True, **kwargs
    ) -> DatasetDict | Dataset:
        """Loads the dataset as a HuggingFace dataset.

        Args:
            splits (list[str], optional): The dataset splits to load.
            retrieve (bool, optional): Whether to retrieve the dataset if it
                does not exist locally. Defaults to True.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (DatasetDict): The loaded dataset.
        """
        return self.load(splits=splits, **kwargs)

    @classmethod
    @abstractmethod
    def can_handle(cls, name: str) -> bool:
        """Checks if the DatasetManager can handle the given dataset.

        Args:
            name (str): The name of the dataset.

        Returns:
            (bool): True if the manager can handle the dataset, False otherwise.
        """
        pass
