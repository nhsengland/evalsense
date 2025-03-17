from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil

from datasets import load_from_disk, Dataset, DatasetDict

from llmscope.constants import PROJECTS_PATH
from llmscope.evaluation import (
    EvaluationResult,
    ResultAggregator,
    ExperimentId,
    ResultId,
)
from llmscope.evaluation.aggregators import DefaultAggregator
from llmscope.utils.huggingface import disable_dataset_progress_bars
from llmscope.utils.files import to_safe_filename


@dataclass
class CachedGenerations:
    experiment_id: ExperimentId
    generations_dataset: Dataset

    def save_to_disk(self, generations_path: Path, exist_ok: bool = False) -> None:
        """Saves the cached generations to a file.

        Args:
            generations_path (Path): The path to save the cached generations.
            exist_ok (bool): Whether to allow overwriting an existing file.
        """
        generations_path.mkdir(parents=True, exist_ok=True)
        filename = to_safe_filename(str(self.experiment_id))
        metadata_file = generations_path / f"{filename}.json"
        if metadata_file.exists() and not exist_ok:
            raise ValueError(
                f"Generations with ID {filename} already exists. "
                "Either choose a different name or set exist_ok=True."
            )

        with disable_dataset_progress_bars():
            self.generations_dataset.save_to_disk(generations_path / "data" / filename)
        with open(metadata_file, "w") as f:
            json.dump(asdict(self.experiment_id), f, indent=4)

    def remove_from_disk(self, generations_path: Path) -> None:
        """Removes the cached generations from disk.

        Args:
            generations_path (Path): The path to the cached generations.
        """
        filename = to_safe_filename(str(self.experiment_id))
        metadata_file = generations_path / f"{filename}.json"
        data_folder = generations_path / "data" / f"{filename}"
        if metadata_file.exists():
            metadata_file.unlink()
        if data_folder.exists():
            shutil.rmtree(data_folder)

    @classmethod
    def load_from_disk(cls, metadata_file_path: Path) -> "CachedGenerations":
        """Loads the cached generations from a file.

        Args:
            metadata_file_path (Path): The path to the metadata file.

        Returns:
            (CachedGenerations): The loaded cached generations.
        """
        generations_path = metadata_file_path.parent

        with open(metadata_file_path, "r") as f:
            json_data = json.load(f)
        experiment_id = ExperimentId(**json_data)

        filename = to_safe_filename(str(experiment_id))
        with disable_dataset_progress_bars():
            generations_dataset = load_from_disk(generations_path / "data" / filename)
        if isinstance(generations_dataset, DatasetDict):
            raise ValueError(
                "Intended to cache a single Dataset, but loaded a DatasetDict. "
                "If you are using a standard Pipeline, please report this issue."
            )

        return cls(experiment_id=experiment_id, generations_dataset=generations_dataset)


@dataclass
class CachedEvaluationResult:
    result_id: ResultId
    evaluation_result: EvaluationResult

    def save_to_disk(self, results_path: Path, exist_ok: bool = False) -> None:
        """Saves the cached evaluation result to a file.

        Args:
            results_path (Path): The path to save the cached evaluation result.
            exist_ok (bool): Whether to allow overwriting an existing file.
        """
        results_path.mkdir(parents=True, exist_ok=True)
        filename = to_safe_filename(str(self.result_id))
        metadata_file = results_path / f"{filename}.json"
        if metadata_file.exists() and not exist_ok:
            raise ValueError(
                f"Result with ID {filename} already exists. "
                "Either choose a different name or set exist_ok=True."
            )

        output_dict = {
            "result_id": asdict(self.result_id),
            "evaluation_result": asdict(self.evaluation_result),
        }
        with open(metadata_file, "w") as f:
            json.dump(output_dict, f, indent=4)

    def remove_from_disk(self, results_path: Path) -> None:
        """Removes the cached evaluation result from disk.

        Args:
            results_path (Path): The path to the cached evaluation result.
        """
        filename = to_safe_filename(str(self.result_id))
        metadata_file = results_path / f"{filename}.json"
        if metadata_file.exists():
            metadata_file.unlink()

    @classmethod
    def load_from_disk(cls, metadata_file_path: Path) -> "CachedEvaluationResult":
        """Loads the cached evaluation result from a file.

        Args:
            metadata_file_path (Path): The path to the metadata file.

        Returns:
            (CachedEvaluationResult): The loaded cached evaluation result.
        """
        with open(metadata_file_path, "r") as f:
            json_data = json.load(f)
        result_id = ResultId(**json_data["result_id"])
        evaluation_result = EvaluationResult(**json_data["evaluation_result"])

        return cls(result_id=result_id, evaluation_result=evaluation_result)


class Project[T]:
    """An LLMScope project, tracking the performed experiments and their results."""

    def __init__(
        self,
        name: str,
        result_aggregator: ResultAggregator[T] | None = None,
        cache_generations: bool = True,
        cache_results: bool = True,
        load_existing: bool = True,
        reset_project: bool = False,
    ) -> None:
        """Initializes a project.

        Args:
            name (str): The name of the project.
            result_aggregator (ResultAggregator | None, optional): The result aggregator
                to use. Defaults to DefaultAggregator if None.
            cache_generations (bool): Whether to cache generations. Defaults to True.
            cache_results (bool): Whether to cache evaluation results. Defaults to True.
            load_existing (bool): Whether to load an existing project if it exists.
                Defaults to True.
            reset_project (bool): Whether to reset the project if it exists. Defaults
                to False. If True, the existing project will be deleted and a new one
                will be created.
        """
        PROJECTS_PATH.mkdir(parents=True, exist_ok=True)
        self.name = name
        if result_aggregator is None:
            result_aggregator = DefaultAggregator()
        self.result_aggregator = result_aggregator
        self.cache_generations = cache_generations
        self.cache_results = cache_results

        if reset_project:
            self.remove()

        project_exists = self.project_path.exists()
        if project_exists and not load_existing:
            raise ValueError(
                f"Project with name {name} already exists. "
                "Either choose a different name or set load_existing=True."
            )
        elif project_exists:
            self._load_existing_project()

    def _load_existing_project(self) -> None:
        """Loads an existing project from disk."""
        for metadata_file in self.generations_path.glob("*.json"):
            cached_generations = CachedGenerations.load_from_disk(metadata_file)
            self.add_generations(
                generations_dataset=cached_generations.generations_dataset,
                experiment_id=cached_generations.experiment_id,
                save_to_disk=False,
                exist_ok=False,
            )

        for metadata_file in self.results_path.glob("*.json"):
            cached_result = CachedEvaluationResult.load_from_disk(metadata_file)
            self.add_result(
                result=cached_result.evaluation_result,
                experiment_id=cached_result.result_id,
                save_to_disk=False,
                exist_ok=False,
            )

    def remove(self) -> None:
        """Removes the project from disk."""
        if self.project_path.exists():
            shutil.rmtree(self.project_path)

    @property
    def project_path(self) -> Path:
        """Returns the path to the project directory."""
        return PROJECTS_PATH / to_safe_filename(self.name)

    @property
    def generations_path(self) -> Path:
        """Returns the path to the generations directory."""
        return self.project_path / "generations"

    @property
    def results_path(self) -> Path:
        """Returns the path to the evaluation results directory."""
        return self.project_path / "results"

    def add_generations(
        self,
        generations_dataset: Dataset,
        experiment_id: ExperimentId,
        save_to_disk: bool = True,
        exist_ok: bool = False,
    ) -> None:
        """Adds generations to the project.

        Args:
            generations_dataset (Dataset): The dataset of generations to add.
            experiment_id (ExperimentId): The ID data of the experiment associated
                with the generations.
            save_to_disk (bool, optional): Whether to save the generations to disk.
                Defaults to True. This flag is ignored if `self.cache_generations` is
                False.
            exist_ok (bool, optional): Specifies whether to allow adding the generations
                with the same ID multiple times when saving to disk (otherwise, the flag
                has no effect). Defaults to False.
        """
        if save_to_disk and self.cache_generations:
            cached_generations = CachedGenerations(
                experiment_id=experiment_id,
                generations_dataset=generations_dataset,
            )
            cached_generations.save_to_disk(self.generations_path, exist_ok=exist_ok)

    def _retrieve_generations(
        self, experiment_id: ExperimentId
    ) -> CachedGenerations | None:
        """Retrieves the generations for a given experiment ID, if they exist.

        Args:
            experiment_id (ExperimentId): The ID data of the experiment.

        Returns:
            (CachedGenerations | None): The cached generations, or None if not found.
        """
        filename = to_safe_filename(str(experiment_id))
        metadata_file = self.generations_path / f"{filename}.json"
        if not metadata_file.exists():
            return None
        cached_generations = CachedGenerations.load_from_disk(metadata_file)
        return cached_generations

    def get_generations(self, experiment_id: ExperimentId) -> Dataset:
        """Retrieves the generations for a given experiment ID, if they exist.

        Args:
            experiment_id (ExperimentId): The ID data of the experiment.

        Returns:
            (Dataset): The dataset of generations, or None if not found.
        """
        cached_generations = self._retrieve_generations(experiment_id)
        if cached_generations is None:
            raise ValueError(
                f"Generations for experiment {experiment_id} do not exist."
            )
        return cached_generations.generations_dataset

    def remove_generations(self, experiment_id: ExperimentId) -> None:
        """Removes the generations for a given experiment ID, if they exist.

        Args:
            experiment_id (ExperimentId): The ID data of the experiment.
        """
        cached_generations = self._retrieve_generations(experiment_id)
        if cached_generations is not None:
            cached_generations.remove_from_disk(self.generations_path)

    def has_generations(self, experiment_id: ExperimentId) -> bool:
        """Checks if generations exist for a given experiment ID.

        Args:
            experiment_id (ExperimentId): The ID data of the experiment.

        Returns:
            (bool): True if generations exist, False otherwise.
        """
        filename = to_safe_filename(str(experiment_id))
        metadata_file = self.generations_path / f"{filename}.json"
        return metadata_file.exists()

    def add_result(
        self,
        result: EvaluationResult,
        experiment_id: ExperimentId,
        save_to_disk: bool = True,
        exist_ok: bool = False,
    ) -> None:
        """Adds a result to the project.

        Args:
            result (EvaluationResult): The evaluation result to add.
            experiment_id (ExperimentId): The ID data of the experiment associated
                with the result.
            save_to_disk (bool, optional): Whether to save the result to disk.
                Defaults to True. This flag is ignored if `self.cache_results` is
                False.
            exist_ok (bool, optional): Specifies whether to allow adding the same
                result multiple times when saving to disk (otherwise, the flag
                has no effect). Defaults to False.
        """
        if save_to_disk and self.cache_results:
            cached_result = CachedEvaluationResult(
                result_id=experiment_id.to_result_id(metric_name=result.name),
                evaluation_result=result,
            )
            cached_result.save_to_disk(self.results_path, exist_ok=exist_ok)
        self.result_aggregator.add_result(
            result=result,
            experiment_id=experiment_id,
            exist_ok=exist_ok,
        )

    def _retrieve_result(self, result_id: ResultId) -> CachedEvaluationResult | None:
        """Retrieves the result for a given experiment ID and metric name, if it exists.

        Args:
            result_id (ResultId): The ID data of the result to retrieve.

        Returns:
            (CachedEvaluationResult | None): The cached evaluation result.
        """
        filename = to_safe_filename(str(result_id))
        metadata_file = self.results_path / f"{filename}.json"
        if not metadata_file.exists():
            return None
        cached_result = CachedEvaluationResult.load_from_disk(metadata_file)
        return cached_result

    def get_result(self, result_id: ResultId) -> EvaluationResult:
        """Retrieves the result for a given experiment ID and metric name.

        Args:
            result_id (ResultId | None): The ID data of the result to retrieve.
                If None, the method will return None.

        Returns:
            (EvaluationResult): The evaluation result, or None if not found.
        """
        cached_result = self._retrieve_result(result_id)
        if cached_result is None:
            raise ValueError(f"Result with ID {result_id} does not exist.")
        return cached_result.evaluation_result

    def remove_result(self, result_id: ResultId) -> None:
        """Removes the result for a given experiment ID and metric name, if it exists.

        Args:
            result_id (ResultId): The ID data of the result to remove.
        """
        cached_result = self._retrieve_result(result_id)
        if cached_result is not None:
            cached_result.remove_from_disk(self.results_path)

    def has_result(self, result_id: ResultId) -> bool:
        """Checks if a result exists for a given experiment ID and metric name.

        Args:
            result_id (ResultId): The ID data of the result to check.

        Returns:
            (bool): True if the result exists, False otherwise.
        """
        filename = to_safe_filename(str(result_id))
        metadata_file = self.results_path / f"{filename}.json"
        return metadata_file.exists()

    def get_result_summary(self) -> T:
        """Returns a summary of the results.

        Returns:
            (T): The summary of the results.
        """
        return self.result_aggregator.return_results()
