from pathlib import Path
from typing import Literal, Optional
from typing_extensions import override
import warnings

from pydantic import BaseModel, field_validator
import yaml

from llmscope.constants import (
    DEFAULT_HASH_TYPE,
    DATASET_CONFIG_PATHS,
)
from llmscope.utils.dict import deep_update
from llmscope.utils.files import to_safe_filename


# TODO: Handle folders
class OnlineSource(BaseModel):
    """The online source of the dataset file(s).

    Attributes:
        url_template (str): The URL template for the dataset file(s),
            optionally taking a version and filename
        requires_auth (bool, optional): Whether accessing the dataset file(s)
            requires authentication
    """

    online: Literal[True]
    url_template: str
    requires_auth: bool = False


class LocalSource(BaseModel):
    """The local source of the dataset file(s).

    Attributes:
        path (str): The path to the dataset file(s)
    """

    online: Literal[False]
    path: Path


class FileMetadata(BaseModel):
    """The metadata for a dataset file.

    Attributes:
        name (str): The name of the dataset file
        hash (str, optional): The hash of the dataset file
        hash_type (str): The type of hash used for the dataset file
        source (OnlineSource | LocalSource, optional): The immediate source of
            the dataset file (use `effective_source` to access the effective source,
            which may be inherited)
        parent (SplitMetadata): The parent split metadata
    """

    name: str
    hash: str | None = None
    hash_type: str = DEFAULT_HASH_TYPE
    source: OnlineSource | LocalSource | None = None
    parent: Optional["SplitMetadata"] = None

    @property
    def effective_source(self) -> OnlineSource | LocalSource:
        """The effective source of the dataset file.

        Returns:
            (OnlineSource | LocalSource): The effective source.
        """
        if self.source is not None:
            return self.source
        return self.parent.effective_source


class SplitMetadata(BaseModel):
    """The metadata for a dataset split.

    Attributes:
        name (str): The name of the dataset split
        files (dict[str, FileMetadata]): The dataset files in the split
        source (OnlineSource | LocalSource, optional): The immediate source of
            the dataset split (use `effective_source` to access the effective source,
            which may be inherited)
        parent (VersionMetadata): The parent version metadata
    """

    name: str
    files: dict[str, FileMetadata]
    source: OnlineSource | LocalSource | None = None
    parent: Optional["VersionMetadata"] = None

    @field_validator("files", mode="before")
    @classmethod
    def convert_list_to_dict(cls, files):
        if isinstance(files, list):
            return {file["name"]: file for file in files}
        return files

    @override
    def model_post_init(self, _):
        for file in self.files.values():
            file.parent = self

    @property
    def effective_source(self) -> OnlineSource | LocalSource:
        """The effective source of the dataset split.

        Returns:
            (OnlineSource | LocalSource): The effective source.
        """
        if self.source is not None:
            return self.source
        return self.parent.effective_source


class VersionMetadata(BaseModel):
    """The metadata for a dataset version.

    Attributes:
        name (str): The name of the dataset version
        splits (dict[str, SplitMetadata], optional): The dataset splits in the version
        files (dict[str, FileMetadata], optional): The dataset files in the version
        source (OnlineSource | LocalSource, optional): The immediate source of
            the dataset version (use `effective_source` to access the effective source,
            which may be inherited)
        parent (DatasetMetadata): The parent dataset metadata
    """

    name: str
    splits: dict[str, SplitMetadata]
    files: dict[str, FileMetadata] | None = None
    source: OnlineSource | LocalSource | None = None
    parent: Optional["DatasetMetadata"] = None

    @field_validator("splits", "files", mode="before")
    @classmethod
    def convert_list_to_dict(cls, vs):
        if isinstance(vs, list):
            return {v["name"]: v for v in vs}
        return vs

    @override
    def model_post_init(self, _):
        for split in self.splits.values():
            split.parent = self

    @property
    def effective_source(self) -> OnlineSource | LocalSource:
        """The effective source of the dataset version.

        Returns:
            (OnlineSource | LocalSource): The effective source.
        """
        if self.source is not None:
            return self.source
        return self.parent.effective_source

    def get_files(self, splits: list[str]) -> dict[str, FileMetadata]:
        """Gets the files for the specified splits.

        Args:
            splits (list[str]): The names of the splits.

        Returns:
            (dict[str, FileMetadata]): The files for the splits.
        """
        files = {}
        if self.files is not None:
            files.update(self.files)
        for split_name in splits:
            if split_name not in self.splits:
                raise ValueError(
                    f"Split '{split_name}' not found for version {self.name}."
                )
            files.update(self.splits[split_name].files)
        return files


class DatasetMetadata(BaseModel):
    """The metadata for a dataset.

    Attributes:
        name (str): The name of the dataset
        versions (dict[str, VersionMetadata]): The dataset versions
        source (OnlineSource | LocalSource, optional): The immediate source of
            the dataset (use `effective_source` to access the effective source,
            which may be inherited)
    """

    name: str
    versions: dict[str, VersionMetadata]
    source: OnlineSource | LocalSource | None = None

    @field_validator("versions", mode="before")
    @classmethod
    def convert_list_to_dict(cls, versions):
        if isinstance(versions, list):
            return {version["name"]: version for version in versions}
        return versions

    @override
    def model_post_init(self, _):
        for version in self.versions.values():
            version.parent = self

    @property
    def effective_source(self) -> OnlineSource | LocalSource:
        """The effective source of the dataset.

        Returns:
            (OnlineSource | LocalSource): The effective source.
        """
        if self.source is not None:
            return self.source
        raise ValueError("No effective source exists.")

    def get_files(self, version: str, splits: list[str]) -> dict[str, FileMetadata]:
        """Gets the files for the specified version and splits.

        Args:
            version (str): The name of the version.
            splits (list[str]): The names of the splits.

        Returns:
            (dict[str, FileMetadata]): The files for the version and splits.
        """
        if version not in self.versions:
            raise ValueError(f"Version '{version}' not found for dataset {self.name}.")
        return self.versions[version].get_files(splits)

    def get_splits(self, version: str) -> dict[str, SplitMetadata]:
        """Gets the dataset splits for the specified version.

        Args:
            version (str): The name of the version.

        Returns:
            (dict[str, SplitMetadata]): The splits for the version.
        """
        if version not in self.versions:
            raise ValueError(f"Version '{version}' not found for dataset {self.name}.")
        return self.versions[version].splits


class DatasetConfig:
    """Configuration for a dataset.

    Attributes:
        dataset_name (str): The name of the dataset.
        dataset_metadata (DatasetMetadata): The metadata for the dataset.
    """

    def __init__(self, dataset_name: str):
        """Initializes a new DatasetConfig.

        Args:
            dataset_name (str): The name of the dataset.
        """
        self.dataset_name = dataset_name
        config = {}
        for config_path in DATASET_CONFIG_PATHS:
            config_file = config_path / (to_safe_filename(dataset_name) + ".yml")
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        new_config = yaml.safe_load(f)
                    config = deep_update(config, new_config)
                except Exception as e:
                    warnings.warn(
                        f"Failed to load dataset config from {config_file}: {e}"
                    )
                    continue
        self.dataset_metadata = DatasetMetadata(**config)

    def get_files(self, version: str, splits: list[str]) -> dict[str, FileMetadata]:
        """Gets the files for the specified version and splits.

        Args:
            version (str): The name of the version.
            splits (list[str]): The names of the splits.

        Returns:
            (dict[str, FileMetadata]): The files for the version and splits.
        """
        return self.dataset_metadata.get_files(version, splits)

    def get_splits(self, version: str) -> dict[str, SplitMetadata]:
        """Gets the dataset splits for the specified version.

        Args:
            version (str): The name of the version.

        Returns:
            (dict[str, SplitMetadata]): The splits for the version.
        """
        return self.dataset_metadata.get_splits(version)
