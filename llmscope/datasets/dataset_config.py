from typing import Iterator
import warnings

import yaml

from llmscope.constants import (
    DEFAULT_VERSION_NAME,
    DEFAULT_HASH_TYPE,
    DATASET_CONFIG_PATHS,
)
from llmscope.datasets.remote_file import RemoteFile
from llmscope.utils.dict import deep_update
from llmscope.utils.files import to_safe_filename


# TODO: Consider using Pydantic for data validation.
class DatasetConfig:
    """Configuration for a dataset."""

    def __init__(self, dataset_name: str):
        """Initializes a new DatasetConfig.

        Args:
            dataset_name (str): The name of the dataset.
        """
        self.dataset_name = dataset_name
        self.config = {}
        for config_path in DATASET_CONFIG_PATHS:
            config_file = config_path / (to_safe_filename(dataset_name) + ".yml")
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        new_config = yaml.safe_load(f)
                    self.update_config(new_config)
                except Exception as e:
                    warnings.warn(
                        f"Failed to load dataset config from {config_file}: {e}"
                    )
                    continue
        if not self.config:
            raise ValueError("No configuration found for dataset {dataset_name}.")

    @property
    def description(self) -> str:
        """The description of the dataset.

        Returns:
            (str): The dataset description.
        """
        return self.config["description"]

    @property
    def all_versions(self) -> list[str]:
        """The available versions of the dataset.

        Returns:
            (list[str]): The available versions.
        """
        return list(self.config["versions"].keys())

    @property
    def all_version_metadata(self) -> dict:
        """The metadata for the available versions of the dataset.

        Returns:
            (dict): The metadata for the available versions.
        """
        return self.config["versions"]

    @property
    def default_version(self) -> str:
        """The default version of the dataset.

        Returns:
            (str): The default version.
        """
        return self.config["default_version"]

    @property
    def default_version_metadata(self) -> dict:
        """The metadata for the default version of the dataset.

        Returns:
            (dict): The metadata for the default version.
        """
        return self.config["versions"][self.default_version]

    def get_remote_files(
        self, version: str = DEFAULT_VERSION_NAME, splits: list[str] | None = None
    ) -> Iterator[RemoteFile]:
        """Gets the remote files to download for a specific dataset.

        Args:
            version (str, optional): The dataset version to retrieve.
            splits (list[str], optional): The dataset splits to retrieve.

        Returns:
            (Iterator[RemoteFile]): The remote files to download.
        """
        source_metadata = {}
        if "source" in self.config:
            source_metadata = self.config["source"]

        if version not in self.all_versions:
            raise ValueError(
                f"Invalid version {version} for dataset {self.dataset_name}."
            )
        version_metadata = self.all_version_metadata[version]

        if "source" in version_metadata:
            source_metadata = deep_update(source_metadata, version_metadata["source"])

        if splits is None:
            splits = list(version_metadata["splits"].keys())
        for split in splits:
            if split not in version_metadata["splits"]:
                raise ValueError(
                    f"Invalid split {split} for dataset {self.dataset_name}."
                )

            split_metadata = version_metadata["splits"][split]

            for filename, file_metadata in split_metadata["files"].items():
                if "source" in file_metadata:
                    source_metadata = deep_update(
                        file_metadata, file_metadata["source"]
                    )

                yield RemoteFile(
                    url=source_metadata["url"].format(
                        version=version, filename=filename
                    ),
                    requires_auth=source_metadata.get("requires_auth", False),
                    filename=filename,
                    expected_size=file_metadata.get("expected_size"),
                    expected_hash=file_metadata.get("expected_hash"),
                    hash_type=file_metadata.get("hash_type", DEFAULT_HASH_TYPE),
                )

    def update_config(self, new_config: dict) -> None:
        """Updates the dataset configuration with data from another config file.

        Args:
            new_config (dict): The new configuration data.
        """
        self.config = deep_update(self.config, new_config)
