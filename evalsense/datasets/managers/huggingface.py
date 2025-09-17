from typing import cast, override

from datasets import DatasetDict, get_dataset_split_names, load_dataset
from huggingface_hub import repo_exists

from evalsense.datasets.dataset_manager import DatasetManager, manager
from evalsense.utils.huggingface import disable_dataset_progress_bars


@manager
class HuggingFaceDatasetManager(DatasetManager):
    """A dataset manager for Hugging Face datasets."""

    priority = 3

    def __init__(
        self,
        name: str,
        version: str = "main",
        splits: list[str] | None = None,
        data_dir: str | None = None,
        **kwargs,
    ):
        """Initializes a new HuggingFaceDatasetManager.

        Args:
            name (str): The name of the dataset.
            version (str, optional): The dataset version to retrieve.
            splits (list[str], optional): The dataset splits to retrieve.
            data_dir (str, optional): The top-level directory for storing all
                datasets. Defaults to "datasets" in the user cache directory.
            **kwargs (dict): Additional keyword arguments.
        """
        if splits is None:
            splits = cast(list[str], get_dataset_split_names(name, revision=version))

        super().__init__(
            name=name,
            version=version,
            splits=splits,
            data_dir=data_dir,
            **kwargs,
        )

    @override
    def retrieve(self, **kwargs) -> None:
        """Downloads and preprocesses a dataset.

        Args:
            **kwargs (dict): Additional keyword arguments.
        """
        dataset = load_dataset(self.name, revision=self.version)
        if not isinstance(dataset, DatasetDict):
            raise ValueError(f"Unexpected dataset type: {type(dataset)}.")
        with disable_dataset_progress_bars():
            dataset.save_to_disk(self.main_data_path)

    @classmethod
    @override
    def can_handle(cls, name: str) -> bool:
        """Checks if the DatasetManager can handle the given dataset.

        Args:
            name (str): The name of the dataset.

        Returns:
            (bool): True if the manager can handle the dataset, False otherwise.
        """
        return repo_exists(name, repo_type="dataset")
