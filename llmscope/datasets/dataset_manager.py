from abc import ABC, abstractmethod


class DatasetManager(ABC):
    """An abstract class for managing datasets."""

    def __init__(self, name: str, priority: int = 0):
        """Initializes a new DatasetManager.

        Args:
            name (str): The name of the dataset.
            priority (int, optional): The priority of the dataset manager when
                choosing between multiple possible managers. Defaults to 0.

        Attributes:
            name (str): The name of the dataset.
            priority (int): The priority of the dataset manager.
        """
        self.name = name
        self.priority = priority

    @abstractmethod
    def get(self, version: str | None = None, splits: list[str] | None = None) -> None:
        """Retrieves and preprocesses a dataset.

        Args:
            version (str, optional): The version of the dataset to retrieve.
            splits (list[str], optional): The splits of the dataset to retrieve.
        """
        pass
