from abc import abstractmethod
from functools import total_ordering
from pathlib import Path
import shutil
from typing import Literal, Protocol, Type, overload, override

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from pydantic import BaseModel

from evalsense.constants import DEFAULT_VERSION_NAME, DATA_PATH
from evalsense.datasets.dataset_config import DatasetConfig, OnlineSource
from evalsense.utils.files import to_safe_filename, download_file


@total_ordering
class DatasetRecord(BaseModel, frozen=True):
    """A record identifying a dataset.

    Attributes:
        name (str): The name of the dataset.
        version (str): The version of the dataset.
        splits (list[str]): The used dataset splits.
    """

    name: str
    version: str
    splits: tuple[str, ...]

    def __eq__(self, other: object) -> bool:
        """Checks if this record is equal to another record.

        Args:
            other (object): The other record to compare with.

        Returns:
            (bool): True if the records are equal, False otherwise.
        """
        if not isinstance(other, DatasetRecord) or type(self) is not type(other):
            return NotImplemented
        return (
            self.name == other.name
            and self.version == other.version
            and self.splits == other.splits
        )

    def __lt__(self, other: object) -> bool:
        """Checks if this record is less than another record.

        Args:
            other (object): The other record to compare with.

        Returns:
            (bool): True if this record is less than the other, False otherwise.
        """
        if not isinstance(other, DatasetRecord) or type(self) is not type(other):
            return NotImplemented
        return (
            self.name,
            self.version,
            self.splits,
        ) < (
            other.name,
            other.version,
            other.splits,
        )

    def __hash__(self) -> int:
        """Returns a hash of the record.

        Returns:
            (int): The hash of the record.
        """
        return hash((self.name, self.version, self.splits))


class DatasetManagerRegistry:
    """A registry for dataset managers."""

    registry: list[Type["DatasetManager"]] = []

    @classmethod
    def register(cls, manager: Type["DatasetManager"]):
        """Registers a new dataset manager.

        Args:
            manager (Type["DatasetManager"]): The dataset manager to be registered.
        """
        cls.registry.append(manager)

    @classmethod
    def get(cls, name: str) -> Type["DatasetManager"] | None:
        """Gets the dataset manager for a specific dataset.

        Args:
            name (str): The name of the dataset.

        Returns:
            (Type["DatasetManager"] | None): The dataset manager for the dataset, or None if not found.
        """
        for manager in sorted(cls.registry, key=lambda m: m.priority, reverse=True):
            if manager.can_handle(name):
                return manager
        return None


def manager(manager: Type["DatasetManager"]) -> Type["DatasetManager"]:
    """Decorator to register a dataset manager.

    Args:
        manager (Type["DatasetManager"]): The dataset manager to register.

    Returns:
        Type["DatasetManager"]: The registered dataset manager.
    """
    DatasetManagerRegistry.register(manager)
    return manager


class DatasetManager(Protocol):
    """A protocol for managing datasets.

    Attributes:
        priority (int): The priority of the dataset manager. Ranges from
            0 (the lowest priority) to 10 (the highest priority).
            A class attribute.
        name (str): The name of the dataset.
        version (str): The used dataset version.
        splits (list[str]): The dataset splits to retrieve.
        data_path (Path): The top-level directory for storing all datasets.
        dataset (Dataset | None): The loaded dataset.
        dataset_dict (DatasetDict | None): The loaded dataset dictionary.
    """

    priority: int = 0

    name: str
    version: str
    splits: list[str]
    data_path: Path
    dataset: Dataset | None
    dataset_dict: DatasetDict | None

    @classmethod
    def create(
        cls,
        name: str,
        splits: list[str],
        version: str | None = None,
        data_dir: str | None = None,
        **kwargs: dict,
    ) -> "DatasetManager":
        """Creates a new dataset manager for the specified dataset.

        Args:
            name (str): The name of the dataset.
            splits (list[str]): The dataset splits to retrieve.
            version (str | None): The dataset version to retrieve.
            data_dir (str | None): The top-level directory for storing all datasets.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            (DatasetManager): The created dataset manager.
        """
        manager = DatasetManagerRegistry.get(name)
        if manager is not None:
            return manager(
                name=name,
                splits=splits,
                version=version,
                data_dir=data_dir,
                **kwargs,
            )
        raise ValueError(f"No suitable dataset manager found for {name}")

    def __init__(
        self,
        name: str,
        splits: list[str],
        version: str | None = None,
        data_dir: str | None = None,
        **kwargs: dict,
    ):
        """Initializes a new DatasetManager.

        Args:
            name (str): The name of the dataset.
            splits (list[str]): The dataset splits to retrieve.
            version (str, optional): The dataset version to retrieve.
            data_dir (str, optional): The top-level directory for storing all
                datasets. Defaults to "datasets" in the user cache directory.
            **kwargs (dict): Additional keyword arguments.
        """
        self.name = name
        self.splits = list(sorted(splits))
        self.version = version or DEFAULT_VERSION_NAME
        if data_dir is not None:
            self.data_path = Path(data_dir)
        else:
            self.data_path = DATA_PATH
        self.dataset = None
        self.dataset_dict = None

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
        """The path for storing the preprocessed dataset files for a specific version.

        Returns:
            (Path): The main dataset directory.
        """
        return self.version_path / "main"

    @property
    def record(self) -> DatasetRecord:
        """Returns a record identifying the dataset.

        Returns:
            (DatasetRecord): The dataset record.
        """
        return DatasetRecord(
            name=self.name,
            version=self.version,
            splits=tuple(self.splits),
        )

    @abstractmethod
    def retrieve(self, **kwargs) -> None:
        """Downloads and preprocesses a dataset.

        Args:
            **kwargs (dict): Additional keyword arguments.
        """
        ...

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

    @overload
    def load(
        self,
        *,
        retrieve: bool = True,
        cache: bool = True,
        force_retrieve: bool = False,
        load_as_dict: Literal[False] = ...,
    ) -> Dataset: ...
    @overload
    def load(
        self,
        *,
        retrieve: bool = True,
        cache: bool = True,
        force_retrieve: bool = False,
        load_as_dict: Literal[True],
    ) -> DatasetDict: ...
    def load(
        self,
        *,
        retrieve: bool = True,
        cache: bool = True,
        force_retrieve: bool = False,
        load_as_dict: bool = False,
    ) -> Dataset | DatasetDict:
        """Loads the dataset as a HuggingFace dataset.

        Args:
            retrieve (bool, optional): Whether to retrieve the dataset if it
                does not exist locally. Defaults to True.
            cache (bool, optional): Whether to cache the dataset in memory.
                Defaults to True.
            force_retrieve (bool, optional): Whether to force retrieving and
                reloading the dataset even if it is already cached. Overrides
                the `retrieve` flag if set to True. Defaults to False.
            load_as_dict (bool, optional): Whether to load the dataset with
                multiple splits as a DatasetDict. If False (the default),
                the selected dataset splits are concatenated into a single
                dataset.

        Returns:
            (Dataset | DatasetDict): The loaded dataset.
        """
        # Return quickly if we already have the dataset cached
        if not load_as_dict and self.dataset is not None and not force_retrieve:
            return self.dataset
        if load_as_dict and self.dataset_dict is not None and not force_retrieve:
            return self.dataset_dict

        # Retrieve the dataset if needed
        if (not self.is_retrieved() and retrieve) or force_retrieve:
            self.retrieve()
        elif not self.is_retrieved():
            raise ValueError(
                f"Dataset {self.name} is not available locally and "
                "retrieve is set to False. Either `retrieve` the dataset first or "
                "set the retrieve flag to True."
            )

        # Load the retrieved dataset
        hf_dataset = load_from_disk(self.main_data_path)
        if not isinstance(hf_dataset, DatasetDict):
            raise ValueError(
                "Expected dataset to be DatasetDict, but got regular Dataset."
            )
        try:
            hf_dataset = DatasetDict({sid: hf_dataset[sid] for sid in self.splits})
        except KeyError as e:
            raise ValueError(f"No such split {e}.")

        if load_as_dict:
            # Return the dataset as a dictionary
            if cache:
                self.dataset_dict = hf_dataset
            return hf_dataset

        # Concatenate the splits and return the data as a single Dataset object
        hf_dataset = concatenate_datasets(
            [
                hf_dataset[s].cast(hf_dataset[self.splits[0]].features)
                for s in self.splits
            ]
        )
        if cache:
            self.dataset = hf_dataset
        return hf_dataset

    def unload(self) -> None:
        """Unloads the dataset from memory."""
        self.dataset = None
        self.dataset_dict = None

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


class FileBasedDatasetManager(DatasetManager):
    """An abstract class for managing datasets.

    Attributes:
        priority (int): The priority of the dataset manager. Ranges from
            0 (the lowest priority) to 10 (the highest priority).
            A class attribute.
        name (str): The name of the dataset.
        version (str): The used dataset version.
        splits (list[str]): The dataset splits to retrieve.
        data_path (Path): The top-level directory for storing all datasets.
        dataset (Dataset | None): The loaded dataset.
        dataset_dict (DatasetDict | None): The loaded dataset dictionary.
        config (DatasetConfig): The configuration for the dataset.
        all_splits: list[str]: All available dataset splits.
    """

    config: DatasetConfig
    all_splits: list[str]

    def __init__(
        self,
        name: str,
        version: str = DEFAULT_VERSION_NAME,
        splits: list[str] | None = None,
        data_dir: str | None = None,
        **kwargs,
    ):
        """Initializes a new DatasetManager.

        Args:
            name (str): The name of the dataset.
            version (str): The dataset version to retrieve.
            splits (list[str], optional): The dataset splits to retrieve.
            data_dir (str, optional): The top-level directory for storing all
                datasets. Defaults to "datasets" in the user cache directory.
            **kwargs (dict): Additional keyword arguments.
        """
        self.config = DatasetConfig(name)
        self.all_splits = list(self.config.get_splits(version).keys())
        if splits is None:
            splits = self.all_splits

        super().__init__(
            name=name,
            version=version,
            splits=splits,
            data_dir=data_dir,
            **kwargs,
        )

    def _retrieve_files(self, **kwargs) -> None:
        """Retrieves  dataset files.

        This method retrieves all the dataset files for the specified splits
        into the `self.version_path` directory.

        Args:
            **kwargs (dict): Additional keyword arguments.
        """
        for filename, file_metadata in self.config.get_files(
            self.version, self.all_splits
        ).items():
            effective_source = file_metadata.effective_source
            if effective_source is not None and isinstance(
                effective_source, OnlineSource
            ):
                download_file(
                    effective_source.url_template.format(
                        version=self.version, filename=filename
                    ),
                    self.version_path / filename,
                    expected_hash=file_metadata.hash,
                    hash_type=file_metadata.hash_type,
                )

    @abstractmethod
    def _preprocess_files(self, **kwargs) -> None:
        """Preprocesses the downloaded dataset files.

        This method preprocesses the retrieved dataset files and saves them
        as a HuggingFace DatasetDict in the `self.main_data_path` directory.

        Args:
            **kwargs (dict): Additional keyword arguments.
        """
        pass

    @override
    def retrieve(self, **kwargs) -> None:
        """Downloads and preprocesses a dataset.

        Args:
            **kwargs (dict): Additional keyword arguments.
        """
        self.version_path.mkdir(parents=True, exist_ok=True)
        self._retrieve_files(**kwargs)
        self._preprocess_files(**kwargs)
