from pathlib import Path
import shutil
from typing import Literal, overload

from inspect_ai.log import EvalLog, read_eval_log
from pydantic import BaseModel, field_serializer, model_validator

from evalsense.constants import PROJECTS_PATH
from evalsense.evaluation import (
    EvaluationRecord,
    GenerationRecord,
    RecordStatus,
    ResultRecord,
)
from evalsense.logging import get_logger
from evalsense.utils.files import to_safe_filename

logger = get_logger(__name__)


class ProjectRecords(BaseModel):
    """Metadata for generation and evaluation records associated with a project."""

    generation: dict[GenerationRecord, ResultRecord] = {}
    evaluation: dict[EvaluationRecord, ResultRecord] = {}

    @field_serializer("generation")
    def serialise_generation(
        self,
        value: dict[GenerationRecord, ResultRecord],
    ) -> list[tuple[dict, dict]]:
        """Convert generation records to a serializable format."""
        return [(k.model_dump(), v.model_dump()) for k, v in value.items()]

    @field_serializer("evaluation")
    def serialise_evaluation(
        self,
        value: dict[EvaluationRecord, ResultRecord],
    ) -> list[tuple[dict, dict]]:
        """Converts evaluation records to a serializable format."""
        return [(k.model_dump(), v.model_dump()) for k, v in value.items()]

    @model_validator(mode="before")
    @classmethod
    def transform_lists_to_dicts(cls, values: dict) -> dict:
        """Converts serialized lists back into dictionaries."""
        values["generation"] = {
            GenerationRecord.model_validate(k): ResultRecord.model_validate(v)
            for k, v in values.get("generation", [])
        }
        values["evaluation"] = {
            EvaluationRecord.model_validate(k): ResultRecord.model_validate(v)
            for k, v in values.get("evaluation", [])
        }
        return values


class Project:
    """An EvalSense project, tracking the performed experiments and their results."""

    METADATA_FILE = "metadata.json"

    def __init__(
        self,
        name: str,
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
        else:
            self.records = ProjectRecords()
            self._save()

    @property
    def project_path(self) -> Path:
        """Returns the path to the project directory."""
        return PROJECTS_PATH / to_safe_filename(self.name)

    @property
    def generation_log_path(self) -> Path:
        """Returns the path to the generation log directory."""
        return self.project_path / "generation_logs"

    @property
    def evaluation_log_path(self) -> Path:
        """Returns the path to the evaluation log directory."""
        return self.project_path / "evaluation_logs"

    def _load_existing_project(self) -> None:
        """Loads an existing project from disk."""
        metadata_file = self.project_path / self.METADATA_FILE
        if not metadata_file.exists():
            raise ValueError(f"Attempting to load a non-existent project {self.name}.")

        with open(metadata_file, "r", encoding="utf-8") as f:
            self.records = ProjectRecords.model_validate_json(f.read())
        self.cleanup_incomplete_logs()

    def _save(self) -> None:
        """Saves the project metadata to disk."""
        self.project_path.mkdir(parents=True, exist_ok=True)
        metadata_file = self.project_path / self.METADATA_FILE
        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write(self.records.model_dump_json(indent=4))

    def remove(self) -> None:
        """Removes the project from disk."""
        if self.project_path.exists():
            shutil.rmtree(self.project_path)

    def _remove_log_file(
        self,
        record: ResultRecord | None,
    ):
        """Removes the log file associated with the record, if it exists.

        Args:
            record (ResultRecord | None): The record associated with the log file.
        """
        if record is not None and record.log_location is not None:
            log_path = Path(record.log_location)
            if log_path.exists():
                log_path.unlink()

    def update_record(
        self,
        record_key: GenerationRecord | EvaluationRecord,
        record_value: ResultRecord,
        *,
        init_eval_record_from_generations: bool = False,
    ):
        """Updates the generation or evaluation record with the specified result.

        Args:
            record_key (GenerationRecord | EvaluationRecord): The generation
                or evaluation record to update.
            record_value (ResultRecord): The generation or evaluation result.
            init_for_evaluation (bool): Whether to initialise a new evaluation
                record if the evaluation record does not exist. Defaults to False.
                This is only applicable if the record_key is an EvaluationRecord.
        """
        current_record = self.get_record(
            record_key,
            init_eval_record_from_generations=init_eval_record_from_generations,
        )
        if (
            current_record is not None
            and current_record.log_location is not None
            and current_record.log_location != record_value.log_location
        ):
            self._remove_log_file(current_record)

        if type(record_key) is GenerationRecord:
            self.records.generation[record_key] = record_value
        elif type(record_key) is EvaluationRecord:
            self.records.evaluation[record_key] = record_value
        else:
            raise TypeError(f"Invalid record type: {type(record_key)}")
        self._save()

    def remove_record(
        self,
        record_key: GenerationRecord | EvaluationRecord,
    ):
        """Removes the generation or evaluation record.

        Args:
            record_key (GenerationRecord | EvaluationRecord): The generation
                or evaluation record to remove.
        """
        if type(record_key) is GenerationRecord:
            record = self.records.generation.pop(record_key, None)
        elif type(record_key) is EvaluationRecord:
            record = self.records.evaluation.pop(record_key, None)
        else:
            raise TypeError(f"Invalid record type: {type(record_key)}")

        self._remove_log_file(record)
        self._save()

    def _retrieve_verify_record(self, record_key: GenerationRecord | EvaluationRecord):
        """Retrieves and verifies the generation or evaluation record.

        Args:
            record_key (GenerationRecord | EvaluationRecord): The generation
                or evaluation record to retrieve.

        Returns:
            ResultRecord | None: The generation or evaluation result, or None if
                a valid record does not exist.
        """
        if type(record_key) is GenerationRecord:
            retrieved_record = self.records.generation.get(record_key, None)
        elif type(record_key) is EvaluationRecord:
            retrieved_record = self.records.evaluation.get(record_key, None)
        else:
            raise TypeError(f"Invalid record type: {type(record_key)}")

        if retrieved_record is not None and retrieved_record.log_location is not None:
            log_path = Path(retrieved_record.log_location)
            if not log_path.exists():
                # Stale record, remove it
                logger.warning(
                    f"⚠️  Log file {log_path} does not exist. Removing stale record."
                )
                self.remove_record(record_key)
                retrieved_record = None
        return retrieved_record

    def get_record(
        self,
        record_key: GenerationRecord | EvaluationRecord,
        *,
        init_eval_record_from_generations: bool = False,
    ) -> ResultRecord | None:
        """Returns the generation or evaluation record for the given key.

        Note: Calling this method may initialise a new evaluation record from
        the matching generation record if the evaluation record does not exist
        yet and `init_eval_record_from_generations` is set to True.

        Args:
            record_key (GenerationRecord | EvaluationRecord): The generation
                or evaluation record to retrieve.
            init_eval_record_from_generations (bool): Whether to initialise a new
                evaluation record if the evaluation record does not exist.
                Defaults to False. This is only applicable if the record_key is
                an EvaluationRecord.

        Returns:
            ResultRecord | None: The generation or evaluation result, or None if
                a valid record does not exist.
        """
        if type(record_key) is GenerationRecord:
            return self._retrieve_verify_record(record_key)
        elif type(record_key) is EvaluationRecord:
            retrieved_eval_record = self._retrieve_verify_record(record_key)
            if (
                retrieved_eval_record is not None
                or not init_eval_record_from_generations
            ):
                return retrieved_eval_record

            generation_result = self._retrieve_verify_record(
                record_key.generation_record
            )
            if generation_result is None:
                return None
            if (
                generation_result.status != "success"
                or generation_result.log_location is None
            ):
                self.records.evaluation[record_key] = generation_result
                self._save()
                return generation_result

            # Create a new evaluation log based on the generation log
            log_path = Path(generation_result.log_location)
            evaluator_name = record_key.evaluator_name
            log_time, core_name, random_id = log_path.stem.split("_", 2)
            new_log_path = self.evaluation_log_path / (
                f"{log_time}_{core_name}-{to_safe_filename(evaluator_name)}_"
                + f"{random_id}{log_path.suffix}"
            )
            new_log_path.parent.mkdir(parents=True, exist_ok=True)
            if not new_log_path.exists():
                shutil.copy(log_path, new_log_path)
            new_record = ResultRecord(
                log_location=str(new_log_path),
            )
            self.records.evaluation[record_key] = new_record
            self._save()
            return new_record
        else:
            raise TypeError(f"Invalid record type: {type(record_key)}")

    def get_log(
        self,
        record_key: GenerationRecord | EvaluationRecord,
        *,
        init_eval_record_from_generations: bool = False,
    ) -> EvalLog | None:
        """Returns the evaluation log for the given record key.

        Args:
            record_key (GenerationRecord | EvaluationRecord): The generation
                or evaluation record to retrieve.
            init_eval_record_from_generations (bool): Whether to initialise a new
                evaluation record if the evaluation record does not exist. Defaults
                to False. This is only applicable if the record_key is an
                EvaluationRecord.

        Returns:
            EvalLog | None: The evaluation log, or None if a valid log does not
                exist.
        """
        record = self.get_record(
            record_key,
            init_eval_record_from_generations=init_eval_record_from_generations,
        )
        if record is not None and record.log_location is not None:
            log_path = Path(record.log_location)
            if log_path.exists():
                return read_eval_log(str(log_path))

    @overload
    def get_logs(
        self,
        type: Literal["generation"],
        status: RecordStatus | None = None,
    ) -> dict[GenerationRecord, EvalLog]: ...
    @overload
    def get_logs(
        self,
        type: Literal["evaluation"],
        status: RecordStatus | None = None,
    ) -> dict[EvaluationRecord, EvalLog]: ...
    def get_logs(
        self,
        type: Literal["generation", "evaluation"],
        status: RecordStatus | None = None,
    ) -> dict[GenerationRecord, EvalLog] | dict[EvaluationRecord, EvalLog]:
        """Returns a dictionary of logs for the given type and status.

        Args:
            type (Literal["generation", "evaluation"]): The type of logs to retrieve.
            status (RecordStatus | None): The status of the logs to retrieve.
                Defaults to None.

        Returns:
            dict[GenerationRecord | EvaluationRecord, EvalLog]: A dictionary of logs.
        """
        if type == "generation":
            records = self.records.generation
        elif type == "evaluation":
            records = self.records.evaluation
        else:
            raise ValueError(f"Invalid log type: {type}")

        results = {}
        for key, value in records.items():
            if status is not None and value.status != status:
                continue
            if value.log_location is not None:
                log_path = Path(value.log_location)
                if log_path.exists():
                    eval_log = read_eval_log(str(log_path))
                    if eval_log is not None:
                        results[key] = eval_log

        return results

    def get_incomplete_logs(
        self,
        type: Literal["generation", "evaluation"],
    ) -> list[EvalLog]:
        """Returns a list of incomplete logs in the project directory.

        Args:
            type (Literal["generation", "evaluation"]): The type of logs to retrieve.

        Returns:
            list[EvalLog]: A list of incomplete logs.
        """
        if type == "generation":
            log_path = self.generation_log_path
            known_logs = [
                v.log_location
                for v in self.records.generation.values()
                if v.log_location
            ]
        elif type == "evaluation":
            log_path = self.evaluation_log_path
            known_logs = [
                v.log_location
                for v in self.records.evaluation.values()
                if v.log_location
            ]
        else:
            raise ValueError(f"Invalid log type: {type}")

        incomplete_logs = []
        extensions = [".json", ".eval"]
        for ext in extensions:
            for log_file in log_path.glob(f"*{ext}"):
                if str(log_file) not in known_logs:
                    loaded_log = read_eval_log(str(log_file))
                    incomplete_logs.append(loaded_log)
        return incomplete_logs

    def cleanup_incomplete_logs(self):
        """Removes all incomplete logs in the project directory."""
        incomplete_logs = self.get_incomplete_logs(
            "generation"
        ) + self.get_incomplete_logs("evaluation")
        for log in incomplete_logs:
            log_path = Path(log.location)
            if log_path.exists():
                log_path.unlink()
