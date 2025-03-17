from dataclasses import dataclass, field

from llmscope.columns import ColumnConfig
from llmscope.datasets import DatasetManager
from llmscope.evaluation import Evaluator
from llmscope.llms import LlmManager
from llmscope.prompts import PromptFormatter
from llmscope.tasks import TaskPreprocessor


@dataclass
class TaskConfig:
    """Configuration for a task to performed by an LLM as part of a pipeline."""

    dataset_manager: DatasetManager
    prompt_formatter: PromptFormatter
    task_preprocessor: TaskPreprocessor | None = None


@dataclass
class ExperimentConfig:
    """Configuration for an experiment to be executed by a pipeline."""

    task: TaskConfig
    llm_manager: LlmManager
    column_config: ColumnConfig = field(default_factory=ColumnConfig)
    evaluator: Evaluator | None = None

    def get_id(self, split_name: str | None = None) -> "ExperimentId":
        """Generates an ID for the experiment.

        Args:
            split_name (str | None): The name of the dataset split, if applicable.

        Returns:
            ExperimentId: An ID for the experiment.
        """
        return ExperimentId(
            dataset_name=self.task.dataset_manager.name,
            prompt_name=self.task.prompt_formatter.name,
            model_name=self.llm_manager.name,
            split_name=split_name,
            task_name=self.task.task_preprocessor.name
            if self.task.task_preprocessor
            else None,
        )


@dataclass
class ExperimentBatchConfig:
    """Configuration for a batch of experiments to be executed by a pipeline."""

    tasks: list[TaskConfig]
    llm_managers: list[LlmManager]
    evaluators: list[Evaluator] | None = None
    column_config: ColumnConfig = field(default_factory=ColumnConfig)

    def validate(self) -> None:
        """Validates the experiment configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.tasks:
            raise ValueError("Experiment must have at least one task.")
        if not self.llm_managers:
            raise ValueError("Experiment must have at least one LLM manager.")

    def all_experiments(self) -> list[ExperimentConfig]:
        """Generates a list of all experiments in the batch.

        Returns:
            list[ExperimentConfig]: A list of all experiments in the batch.
        """
        experiments = []
        for task in self.tasks:
            for llm_manager in self.llm_managers:
                for evaluator in self.evaluators or [None]:
                    experiments.append(
                        ExperimentConfig(
                            task=task,
                            llm_manager=llm_manager,
                            evaluator=evaluator,
                            column_config=self.column_config,
                        )
                    )
        return experiments


@dataclass(frozen=True, kw_only=True)
class ExperimentId:
    """Data identifying an experiment.

    Attributes:
        dataset_name (str): The name of the dataset.
        prompt_name (str): The name of the prompt.
        model_name (str): The name of the model.
        split_name (str | None): The name of the dataset split, if applicable.
        task_name (str | None): The name of the task.
    """

    dataset_name: str
    prompt_name: str
    model_name: str
    split_name: str | None = None
    task_name: str | None = None

    def to_result_id(self, metric_name: str) -> "ResultId":
        """Generates a unique result ID for the experiment.

        Args:
            metric_name (str): The name of the metric.

        Returns:
            str: A unique identifier for the experiment result.
        """
        return ResultId(
            dataset_name=self.dataset_name,
            prompt_name=self.prompt_name,
            model_name=self.model_name,
            split_name=self.split_name,
            task_name=self.task_name,
            metric_name=metric_name,
        )

    def __str__(self) -> str:
        """Generates a string representation of the experiment ID.

        Returns:
            str: A string representation of the experiment ID.
        """
        return (
            f"{self.dataset_name}-{self.prompt_name}-{self.model_name}-"
            f"{self.split_name or ''}-{self.task_name or ''}"
        ).strip("-")


@dataclass(frozen=True, kw_only=True)
class ResultId(ExperimentId):
    """Data identifying an experimental result.

    Attributes:
        dataset_name (str): The name of the dataset.
        prompt_name (str): The name of the prompt.
        model_name (str): The name of the model.
        split_name (str | None): The name of the dataset split, if applicable.
        task_name (str | None): The name of the task.
        metric_name (str): The name of the metric.
    """

    metric_name: str

    def to_experiment_id(self) -> ExperimentId:
        """Generates an ExperimentId object from the ResultId.

        Returns:
            ExperimentId: An ExperimentId object with the same attributes as the
                result ID, except for metric_name.
        """
        return ExperimentId(
            dataset_name=self.dataset_name,
            prompt_name=self.prompt_name,
            model_name=self.model_name,
            split_name=self.split_name,
            task_name=self.task_name,
        )

    def __str__(self) -> str:
        """Generates a string representation of the result ID.

        Returns:
            str: A string representation of the result ID.
        """
        experiment_id_str = super().__str__()
        return f"{experiment_id_str}-{self.metric_name}"
