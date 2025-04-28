from abc import abstractmethod
from typing import Protocol

from llmscope.workflow.project import Project


class ResultAnalyser[T](Protocol):
    """A protocol for analysing or aggregating evaluation results.

    This class is generic in T to enable returning different types of results.
    """

    name: str

    def __init__(self, name: str) -> None:
        """Initializes the result analyser.

        Args:
            name (str): The name of the result analyser.
        """
        self.name = name

    @abstractmethod
    def __call__(self, project: Project, **kwargs: dict) -> T:
        """Analyses the evaluation results.

        Args:
            project (Project): The project holding the evaluation data to analyse.
            **kwargs (dict): Additional arguments for the analysis.
        """
        ...
