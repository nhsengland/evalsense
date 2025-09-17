from typing_extensions import override

from datasets import Dataset, DatasetDict
import polars as pl

from evalsense.datasets.dataset_manager import (
    FileBasedDatasetManager,
    manager,
)
from evalsense.utils.huggingface import disable_dataset_progress_bars

_DEFAULT_VERSION = "5d3cd4d8a25b4ebb5b2b87c3923a7b2b7150e33d"


@manager
class AciBenchDatasetManager(FileBasedDatasetManager):
    """A dataset manager for the ACI Bench dataset."""

    _DATASET_NAME = "ACI-BENCH"
    priority = 7

    def __init__(
        self,
        version: str | None = _DEFAULT_VERSION,
        splits: list[str] | None = None,
        data_dir: str | None = None,
        **kwargs,
    ):
        """Initializes a new AciBenchDatasetManager.

        Args:
            version (str, optional): The dataset version to retrieve.
            splits (list[str], optional): The dataset splits to retrieve.
            data_dir (str, optional): The top-level directory for storing all
                datasets. Defaults to "datasets" in the user cache directory.
            **kwargs (dict): Additional keyword arguments.
        """
        kwargs.pop("name", None)
        if version is None:
            version = _DEFAULT_VERSION
        super().__init__(
            name=self._DATASET_NAME,
            version=version,
            splits=splits,
            data_dir=data_dir,
            **kwargs,
        )

    @override
    def _preprocess_files(self, **kwargs) -> None:
        """Preprocesses the downloaded dataset files.

        This method preprocesses the downloaded dataset files and saves them
        as a HuggingFace DatasetDict in the `self.main_data_path` directory.

        Args:
            **kwargs (dict): Additional keyword arguments.
        """
        dataset_dict = {}
        for split in self.all_splits:
            # Join all data files into a single DataFrame
            data_df = None
            for file in self.config.get_files(self.version, [split]).values():
                if data_df is None:
                    data_df = pl.read_csv(self.version_path / file.name)
                else:
                    other_df = pl.read_csv(self.version_path / file.name)
                    data_df = data_df.join(
                        other_df, on=["dataset", "encounter_id"], how="inner"
                    )
            if data_df is None:
                raise RuntimeError(f"No data found for split '{split}'.")
            dataset = Dataset.from_polars(data_df)
            dataset_dict[split] = dataset

        # Save the dataset to disk
        with disable_dataset_progress_bars():
            hf_dataset = DatasetDict(dataset_dict)
            hf_dataset.save_to_disk(self.main_data_path)

    @classmethod
    @override
    def can_handle(cls, name: str) -> bool:
        """Checks if the DatasetManager can handle the given dataset.

        Args:
            name (str): The name of the dataset.

        Returns:
            (bool): True if the manager can handle the dataset, False otherwise.
        """
        return name.lower() == cls._DATASET_NAME.lower()
