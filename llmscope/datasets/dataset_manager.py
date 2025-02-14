from abc import ABC, abstractmethod
from pathlib import Path
import shutil

from llmscope.constants import DATA_PATH
from llmscope.utils import to_safe_filename


class DatasetManager(ABC):
    """An abstract class for managing datasets."""

    _DEFAULT_VERSION_DIR_NAME = "default"

    def __init__(
        self, name: str, priority: int = 0, data_dir: str | None = None, **kwargs
    ):
        """Initializes a new DatasetManager.

        Args:
            name (str): The name of the dataset.
            priority (int, optional): The priority of the dataset manager when
                choosing between multiple possible managers. Recommended values
                range from 0 to 10, with 0 (the lowest) being the default.
            data_dir (str, optional): The top-level directory for storing all
                datasets. Defaults to "datasets" in the user cache directory.

        Attributes:
            name (str): The name of the dataset.
            priority (int): The priority of the dataset manager.
            data_path (Path): The top-level directory for storing all datasets.
        """
        self.name = name
        self.priority = priority
        if data_dir is not None:
            self.data_path = Path(data_dir)
        else:
            self.data_path = DATA_PATH
        self.data_path.mkdir(parents=True, exist_ok=True)

    @property
    def dataset_path(self) -> Path:
        """The directory for storing this dataset.

        Returns:
            (Path): The dataset directory.
        """
        return self.data_path / to_safe_filename(self.name)

    def get_version_path(self, version: str | None = None) -> Path:
        """Returns the directory for storing a specific version of this dataset.

        Args:
            version (str): The dataset version.

        Returns:
            (Path): The dataset version directory.
        """
        if version is None:
            return self.dataset_path / self._DEFAULT_VERSION_DIR_NAME
        return self.dataset_path / to_safe_filename(version)

    @abstractmethod
    def _get_files(
        self, version: str | None = None, splits: list[str] | None = None, **kwargs
    ) -> None:
        """Downloads and preprocesses dataset files.

        Args:
            version (str, optional): The dataset version to retrieve.
            splits (list[str], optional): The dataset splits to retrieve.
        """
        pass

    def get(
        self, version: str | None = None, splits: list[str] | None = None, **kwargs
    ) -> None:
        """Downloads and preprocesses a dataset.

        Args:
            version (str, optional): The dataset version to retrieve.
            splits (list[str], optional): The dataset splits to retrieve.
        """
        self._get_files(version, splits, **kwargs)

    def is_downloaded(self, version: str | None = None) -> bool:
        """Checks if the dataset is already downloaded.

        Args:
            version (str, optional): The dataset version to check.

        Returns:
            (bool): True if the dataset exists locally, False otherwise.
        """
        return self.get_version_path(version).exists()

    def remove(self, version: str | None = None) -> None:
        """Deletes the dataset from disk.

        Args:
            version (str, optional): The dataset version to remove.
        """
        version_path = self.get_version_path(version)
        if version_path.exists():
            shutil.rmtree(version_path)

    @abstractmethod
    def load(self, version: str | None = None, splits: list[str] | None = None):
        """Loads the dataset with task-specific preprocessing.

        Args:
            version (str, optional): The dataset version to load.
            splits (list[str], optional): The dataset splits to load.

        Returns:
            (DatasetDict | Dataset): The loaded dataset.
        """
        pass

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
