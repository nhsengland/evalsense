from typing_extensions import override

from datasets import Dataset, DatasetDict
import polars as pl

from llmscope.datasets import DatasetManager
from llmscope.utils.huggingface import disable_dataset_progress_bars


class AciBenchDatasetManager(DatasetManager):
    """A dataset manager for the ACI Bench dataset."""

    _DATASET_NAME = "ACI-BENCH"

    def __init__(
        self,
        version: str = "5d3cd4d8a25b4ebb5b2b87c3923a7b2b7150e33d",
        data_dir: str | None = None,
        **kwargs,
    ):
        """Initializes a new AciBenchDatasetManager.

        Args:
            version (str, optional): The dataset version to retrieve.
            data_dir (str, optional): The top-level directory for storing all
                datasets. Defaults to "datasets" in the user cache directory.
            **kwargs (dict): Additional keyword arguments.
        """
        super().__init__(
            self._DATASET_NAME, version=version, priority=7, data_dir=data_dir, **kwargs
        )

    @override
    def _preprocess_files(self, splits: list[str], **kwargs) -> None:
        """Preprocesses the downloaded dataset files.

        This method preprocesses the downloaded dataset files and saves them
        as a HuggingFace DatasetDict in the `self.main_data_path` directory.

        Args:
            splits (list[str]): The dataset splits to preprocess.
            **kwargs (dict): Additional keyword arguments.
        """
        dataset_dict = {}
        for split in splits:
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
        return name == cls._DATASET_NAME
