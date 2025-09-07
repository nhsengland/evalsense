from dataclasses import dataclass, field
from functools import total_ordering
from typing import Literal

from inspect_ai.dataset import FieldSpec, RecordToSample
from pydantic import BaseModel

from evalsense.datasets import DatasetManager, DatasetRecord
from evalsense.evaluation import Evaluator
from evalsense.generation import GenerationSteps, ModelConfig, ModelRecord
from evalsense.tasks import DefaultTaskPreprocessor, TaskPreprocessor

type RecordStatus = Literal["started", "success", "cancelled", "error"]
type ExperimentDefinitions = (
    ExperimentConfig
    | ExperimentBatchConfig
    | list[ExperimentConfig]
    | list[ExperimentBatchConfig]
    | list[ExperimentConfig | ExperimentBatchConfig]
)


class ResultRecord(BaseModel, frozen=True):
    """A record indicating the result of generation or evaluation.

    Attributes:
        status (RecordStatus): The status of the record.
        error_message (str | None): The error message, if any.
        log_location (str | None): The location of the associated Inspect log file.
    """

    status: RecordStatus = "started"
    error_message: str | None = None
    log_location: str | None = None


@total_ordering
class GenerationRecord(BaseModel, frozen=True):
    """A record identifying generations for a specific task.

    Attributes:
        dataset_record (DatasetRecord): The record of the dataset.
        generator_name (str): The name of the generator.
        task_name (str): The name of the task.
        model_record (ModelRecord): The record of the model.
        experiment_name (str | None): The name of the experiment, if applicable.
    """

    dataset_record: DatasetRecord
    generator_name: str
    task_name: str
    model_record: ModelRecord
    experiment_name: str | None = None

    def get_evaluation_record(self, evaluator_name: str) -> "EvaluationRecord":
        """Generates an evaluation record from the generation record.

        Args:
            evaluator_name (str): The name of the evaluator.

        Returns:
            EvaluationRecord: The evaluation record.
        """
        return EvaluationRecord(
            **self.model_dump(),
            evaluator_name=evaluator_name,
        )

    @property
    def label(self) -> str:
        """Generates a label for the generation record.

        Returns:
            str: The label for the generation record.
        """
        return (
            f"{self.dataset_record.name} | {self.task_name} | "
            f"{self.generator_name} | {self.model_record.name}"
        )

    def __eq__(self, other: object) -> bool:
        """Checks equality with another record.

        Args:
            other (object): The other record to compare with.

        Returns:
            bool: True if the records are equal, False otherwise.
        """
        if not isinstance(other, GenerationRecord) or type(self) is not type(other):
            return NotImplemented
        return (
            self.dataset_record == other.dataset_record
            and self.generator_name == other.generator_name
            and self.task_name == other.task_name
            and self.model_record == other.model_record
            and self.experiment_name == other.experiment_name
        )

    def __lt__(self, other: object) -> bool:
        """Checks if this record is less than another record.

        Args:
            other (object): The other record to compare with.

        Returns:
            bool: True if this record is less than the other, False otherwise.
        """
        if not isinstance(other, GenerationRecord) or type(self) is not type(other):
            return NotImplemented
        return (
            self.dataset_record,
            self.generator_name,
            self.task_name,
            self.model_record,
            self.experiment_name or "",
        ) < (
            other.dataset_record,
            other.generator_name,
            other.task_name,
            other.model_record,
            other.experiment_name or "",
        )

    def __hash__(self) -> int:
        """Generates a hash for the generation record.

        Returns:
            int: The hash of the generation record.
        """
        return hash(
            (
                self.dataset_record,
                self.generator_name,
                self.task_name,
                self.model_record,
                self.experiment_name,
            )
        )


@total_ordering
class EvaluationRecord(GenerationRecord, frozen=True):
    """A record identifying evaluations for a specific task.

    Attributes:
        dataset_record (DatasetRecord): The record of the dataset.
        generator_name (str): The name of the generator.
        task_name (str): The name of the task.
        model_record (ModelRecord): The record of the model.
        experiment_name (str | None): The name of the experiment, if applicable.
        evaluator_name (str): The name of the evaluator.
    """

    evaluator_name: str

    @property
    def generation_record(self) -> GenerationRecord:
        """Generates a generation record from the evaluation record.

        Returns:
            GenerationRecord: The generation record.
        """
        return GenerationRecord(
            **self.model_dump(exclude={"evaluator_name"}),
        )

    def get_meta_grouped_record(self, metric_name: str) -> "MetaTierGroupedRecord":
        """Generates a perturbation grouped record from the evaluation record.

        Args:
            metric_name (str): The name of the metric being evaluated.

        Returns:
            PerturbationGroupedRecord: The perturbation grouped record.
        """
        return MetaTierGroupedRecord(
            **self.model_dump(exclude={"generator_name"}),
            generator_name="",
            metric_name=metric_name,
        )

    @property
    def label(self) -> str:
        """Generates a label for the evaluation record.

        Returns:
            str: The label for the evaluation record.
        """
        return (
            f"{self.dataset_record.name} | {self.task_name} | {self.generator_name} | "
            f"{self.model_record.name} | {self.evaluator_name}"
        )

    def __eq__(self, other: object) -> bool:
        """Checks equality with another record.

        Args:
            other (object): The other record to compare with.

        Returns:
            bool: True if the records are equal, False otherwise.
        """
        if not isinstance(other, EvaluationRecord) or type(self) is not type(other):
            return NotImplemented
        generation_equal = super().__eq__(other)
        if generation_equal is NotImplemented:
            return generation_equal
        return generation_equal and self.evaluator_name == other.evaluator_name

    def __lt__(self, other: object) -> bool:
        """Checks if this record is less than another record.

        Args:
            other (object): The other record to compare with.

        Returns:
            bool: True if this record is less than the other, False otherwise.
        """
        if not isinstance(other, EvaluationRecord) or type(self) is not type(other):
            return NotImplemented
        if super().__lt__(other):
            return True
        elif super().__eq__(other):
            return self.evaluator_name < other.evaluator_name
        else:
            return False

    def __hash__(self) -> int:
        """Generates a hash for the evaluation record.

        Returns:
            int: The hash of the evaluation record.
        """
        return hash(
            (
                self.dataset_record,
                self.generator_name,
                self.task_name,
                self.model_record,
                self.experiment_name,
                self.evaluator_name,
            )
        )


@total_ordering
class MetaTierGroupedRecord(EvaluationRecord, frozen=True):
    """A record grouping evaluation records by generator name
    (as generator name specifies the meta tier, which defines
    the expected score ranking).

    Attributes:
        dataset_record (DatasetRecord): The record of the dataset.
        generator_name (str): The name of the generator.
        task_name (str): The name of the task.
        model_record (ModelRecord): The record of the model.
        experiment_name (str | None): The name of the experiment, if applicable.
        evaluator_name (str): The name of the evaluator.
        metric_name (str): The name of the metric being evaluated.
    """

    metric_name: str

    def __eq__(self, other: object) -> bool:
        """Checks equality with another record.

        Args:
            other (object): The other record to compare with.

        Returns:
            bool: True if the records are equal, False otherwise.
        """
        if not isinstance(other, MetaTierGroupedRecord) or type(self) is not type(
            other
        ):
            return NotImplemented
        generation_equal = super().__eq__(other)
        if generation_equal is NotImplemented:
            return generation_equal
        return generation_equal and self.metric_name == other.metric_name

    def __lt__(self, other: object) -> bool:
        """Checks if this record is less than another record.

        Args:
            other (object): The other record to compare with.

        Returns:
            bool: True if this record is less than the other, False otherwise.
        """
        if not isinstance(other, MetaTierGroupedRecord) or type(self) is not type(
            other
        ):
            return NotImplemented
        if super().__lt__(other):
            return True
        elif super().__eq__(other):
            return self.metric_name < other.metric_name
        else:
            return False

    def __hash__(self) -> int:
        """Generates a hash for the perturbation grouped record.

        Returns:
            int: The hash of the perturbation grouped record.
        """
        return hash(
            (
                self.dataset_record,
                self.generator_name,
                self.task_name,
                self.model_record,
                self.experiment_name,
                self.evaluator_name,
                self.metric_name,
            )
        )


@dataclass
class ExperimentConfig:
    """Configuration for an experiment to be executed by a pipeline."""

    dataset_manager: DatasetManager
    generation_steps: GenerationSteps
    model_config: ModelConfig
    field_spec: FieldSpec | RecordToSample | None = None
    task_preprocessor: TaskPreprocessor = field(
        default_factory=lambda: DefaultTaskPreprocessor()
    )
    evaluator: Evaluator | None = None
    name: str | None = None

    @property
    def generation_record(self) -> GenerationRecord:
        """A identifying generations for a specific task.

        Returns:
            GenerationsRecord: A record of the generations for the experiment.
        """
        return GenerationRecord(
            dataset_record=self.dataset_manager.record,
            generator_name=self.generation_steps.name,
            task_name=self.task_preprocessor.name,
            model_record=self.model_config.record,
            experiment_name=self.name,
        )

    @property
    def evaluation_record(self) -> EvaluationRecord:
        """A identifying evaluations for a specific task.

        Returns:
            EvaluationRecord: A record of the evaluations for the experiment.
        """
        if self.evaluator is None:
            raise ValueError(
                "Cannot get evaluation record for an experiment without an evaluator"
            )

        return self.generation_record.get_evaluation_record(
            self.evaluator.name,
        )


@dataclass
class TaskConfig:
    """Configuration for a task to be executed by a pipeline."""

    dataset_manager: DatasetManager
    generation_steps: GenerationSteps
    field_spec: FieldSpec | RecordToSample | None = None
    task_preprocessor: TaskPreprocessor = field(
        default_factory=lambda: DefaultTaskPreprocessor()
    )


@dataclass
class ExperimentBatchConfig:
    """Configuration for a batch of experiments to be executed by a pipeline."""

    tasks: list[TaskConfig]
    model_configs: list[ModelConfig]
    evaluators: list[Evaluator] = field(default_factory=list)
    name: str | None = None

    def validate(self) -> None:
        """Validates the experiment configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.tasks:
            raise ValueError("Experiment must have at least one task.")
        if not self.model_configs:
            raise ValueError("Experiment must have at least one LLM manager.")

    @property
    def all_experiments(self) -> list[ExperimentConfig]:
        """Generates a list of all experiments in the batch.

        Returns:
            list[ExperimentConfig]: A list of all experiments in the batch.
        """
        experiments = []
        for task in self.tasks:
            for llm_manager in self.model_configs:
                if self.evaluators:
                    for evaluator in self.evaluators:
                        experiments.append(
                            ExperimentConfig(
                                dataset_manager=task.dataset_manager,
                                generation_steps=task.generation_steps,
                                field_spec=task.field_spec,
                                task_preprocessor=task.task_preprocessor,
                                model_config=llm_manager,
                                evaluator=evaluator,
                                name=self.name,
                            )
                        )
                else:
                    experiments.append(
                        ExperimentConfig(
                            dataset_manager=task.dataset_manager,
                            generation_steps=task.generation_steps,
                            field_spec=task.field_spec,
                            task_preprocessor=task.task_preprocessor,
                            model_config=llm_manager,
                            name=self.name,
                        )
                    )
        return experiments
