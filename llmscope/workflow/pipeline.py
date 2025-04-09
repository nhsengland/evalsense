import os
from typing import Any, cast

from inspect_ai import Task, eval, eval_retry, score, task
from inspect_ai.dataset import Dataset
from inspect_ai.log import EvalLog, read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, Model, get_model
import psutil
from tqdm.auto import tqdm

from llmscope.evaluation import (
    Evaluator,
    ExperimentConfig,
    ExperimentBatchConfig,
    ResultRecord,
    ScorerFactory,
)

from llmscope.logging import get_logger
from llmscope.generation import ModelConfig
from llmscope.utils.files import to_safe_filename
from llmscope.workflow.project import Project

type ExperimentDefinitions = (
    ExperimentConfig
    | ExperimentBatchConfig
    | list[ExperimentConfig | ExperimentBatchConfig]
)

logger = get_logger(__name__)


@task
def create_task(experiment: ExperimentConfig, dataset: Dataset) -> Task:
    """Creates an Inspect AI task for the experiment.

    Args:
        experiment (ExperimentConfig): The experiment configuration.
        dataset (Dataset): The dataset to process.
    """
    return Task(
        dataset=dataset,
        solver=experiment.generation_steps.solver,
        name=to_safe_filename(experiment.generation_record.label),
    )


class Pipeline[T]:
    """A pipeline for evaluating LLMs."""

    def __init__(
        self,
        experiments: ExperimentDefinitions,
        project: Project[T],
        maintain_order: bool = False,
    ):
        """Initializes a new Pipeline.

        Args:
            experiments (ExperimentBatchConfig | ExperimentConfig
                | list[ExperimentBatchConfig | ExperimentConfig]): The experiments
                to run in the pipeline.
            project (Project): The project in which to track the results and outputs.
            maintain_order (bool): Whether to maintain the order of the experiments or
                whether to reorder them to reduce the number of model loads. Defaults
                to False.
        """
        # Standardize experiments to a list of ExperimentConfigs
        if not isinstance(experiments, list):
            experiments = [experiments]
        all_experiments: list[ExperimentConfig] = []
        for experiment in experiments:
            if isinstance(experiment, ExperimentBatchConfig):
                experiment.validate()
                all_experiments.extend(experiment.all_experiments)
            else:
                all_experiments.append(experiment)
        if not maintain_order:
            all_experiments = sorted(all_experiments, key=lambda x: x.model_config.name)
        self.experiments = all_experiments
        self.project = project
        self._active_model_config: ModelConfig | None = None
        self._active_model: Model | None = None

    def _cleanup_active_model(self):
        """Cleans up the active model if it exists."""
        if self._active_model is not None:
            # Cleanup background processes to free CUDA memory
            # Temporary workaround for Inspect + vLLM memory leaks,
            # see https://github.com/UKGovernmentBEIS/inspect_ai/issues/1543
            main_id = os.getpid()
            parent = psutil.Process(main_id)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, still_alive = psutil.wait_procs(children, timeout=5)
            if still_alive:
                logger.warning("⚠️  Failed to clean up background processes.")

            self._active_model_config = None
            self._active_model = None

    def _load_model(
        self,
        new_model_config: ModelConfig,
    ) -> Model:
        """Gets the model for the current experiment.

        Args:
            new_model_config (ModelConfig): The model configuration for the new model
                to be loaded.

        Returns:
            Model: The model for the current experiment.
        """
        if new_model_config != self._active_model_config:
            logger.info(f"▶️  Loading model {new_model_config.name}.")

            # Loading a new model — clean up the previous one
            self._cleanup_active_model()

            # Prepare the new model
            if isinstance(new_model_config.model, Model):
                new_model = new_model_config.model
            else:
                new_model = get_model(
                    model=new_model_config.model,
                    **new_model_config.model_args,
                    config=GenerateConfig(**new_model_config.generation_args),
                )

            self._active_model_config = new_model_config
            self._active_model = new_model

            return new_model

        # Reusing the previous model
        return cast(Model, self._active_model)

    def _generate_on_dataset(
        self,
        experiment: ExperimentConfig,
        inspect_dataset: Dataset,
        force_rerun: bool,
        eval_kwargs: dict[str, Any] | None,
        eval_retry_kwargs: dict[str, Any] | None,
    ):
        """Generates the results for a given dataset and experiment.

        Args:
            experiment (ExperimentConfig): The experiment configuration.
            inspect_dataset (Dataset): The dataset to process.
            force_rerun (bool): Whether to force rerun the experiment.
            eval_kwargs (dict[str, Any], optional): Additional arguments to pass
                to the Inspect eval function. Defaults to empty dictionary when
                None.
            eval_retry_kwargs (dict[str, Any], optional): Additional arguments
                to pass to the Inspect eval function for retrying failed tasks.
                Defaults to empty dictionary when None.
        """
        prev_record = self.project.get_record(experiment.generation_record)
        interrupted = False
        # We need to create the task even when resuming from a previous log,
        # otherwise Inspect will not be able to resolve it.
        task = create_task(
            experiment=experiment,
            dataset=inspect_dataset,
        )
        # TODO: Full implementation of resuming from logs is currently blocked
        # by issues with Inspect AI. See:
        # #1556 https://github.com/UKGovernmentBEIS/inspect_ai/issues/1556
        # #1565 https://github.com/UKGovernmentBEIS/inspect_ai/issues/1565
        RESUMING_FROM_LOG_NOT_YET_IMPLEMENTED = True
        if (
            prev_record is None
            or prev_record.log_location is None
            or force_rerun
            or RESUMING_FROM_LOG_NOT_YET_IMPLEMENTED
        ):
            self.project.remove_record(experiment.generation_record)
            self.project.update_record(experiment.generation_record, ResultRecord())
            try:
                eval_logs = eval(
                    tasks=task,
                    model=self._active_model,
                    log_dir=str(self.project.generation_log_path),
                    score=False,
                    **(eval_kwargs or dict()),
                )
            except BaseException as e:
                eval_logs = self.project.get_incomplete_logs(type="generation")
                interrupted = isinstance(e, KeyboardInterrupt)
        else:
            logger.info(
                f"🔁  Retrying generation using log: {prev_record.log_location}"
            )
            prev_log = read_eval_log(prev_record.log_location)
            try:
                eval_logs = eval_retry(
                    tasks=prev_log,
                    log_dir=str(self.project.generation_log_path),
                    **(eval_retry_kwargs or dict()),
                )
            except BaseException as e:
                eval_logs = self.project.get_incomplete_logs(type="generation")
                interrupted = isinstance(e, KeyboardInterrupt)

        status = "error"
        error_message = "Unknown error"
        log_location = None
        if not eval_logs:
            error_message = "No log returned from an experiment."
            logger.error("❌  Generation failed: no log returned from an experiment.")
        else:
            if len(eval_logs) > 1:
                logger.warning(
                    f"⚠️  Unexpected number of eval logs ({len(eval_logs)} > 1), "
                    "results may be ignored."
                )
            eval_log = eval_logs[0]
            log_location = eval_log.location

            if eval_log.status == "error":
                if eval_log.error is not None:
                    error_message = eval_log.error.message
                logger.error(f"❌  Generation failed due to an error: {error_message}")
            elif eval_log.status == "cancelled":
                error_message = "Generation was cancelled."
                logger.error("❌  Generation was cancelled.")
            elif eval_log.status == "success":
                status = "success"
                error_message = None
                logger.info(
                    f"✅  Generation for {experiment.generation_record.label} "
                    "completed successfully."
                )
        self.project.update_record(
            experiment.generation_record,
            ResultRecord(
                status=status, error_message=error_message, log_location=log_location
            ),
        )

        if interrupted:
            logger.critical("🛑  Execution was interrupted.")
            raise KeyboardInterrupt()

    def generate(
        self,
        show_progress: bool = True,
        force_rerun: bool = False,
        force_reload: bool = False,
        eval_kwargs: dict[str, Any] | None = None,
        eval_retry_kwargs: dict[str, Any] | None = None,
    ):
        """Runs the generation stage of the pipeline.

        Args:
            show_progress (bool, optional): Whether to show a progress bar.
                Defaults to True.
            force_rerun (bool, optional): Whether to force rerun the experiments.
                Defaults to False.
            force_reload (bool, optional): Whether to force reloading and
                reprocessing the datasets. Defaults to False.
            eval_kwargs (dict[str, Any], optional): Additional arguments to pass
                to the Inspect eval function. Defaults to empty dictionary when
                None.
            eval_retry_kwargs (dict[str, Any], optional): Additional arguments
                to pass to the Inspect eval function for retrying failed tasks.
                Defaults to empty dictionary when None.
        """
        for experiment in tqdm(
            self.experiments, disable=not show_progress, desc="Experiment Generation"
        ):
            logger.info(
                f"🔄  Starting generation for {experiment.generation_record.label}"
            )
            prev_record = self.project.get_record(
                experiment.generation_record,
            )
            if prev_record is not None and prev_record.status == "success":
                logger.info("⏭️  Generation skipped — already completed.")
                continue

            logger.info(f"▶️  Loading dataset {experiment.dataset_manager.name}.")
            dataset_manager = experiment.dataset_manager
            hf_dataset = dataset_manager.load(
                retrieve=not force_reload,
                cache=True,
                force_retrieve=force_reload,
            )

            logger.info(
                "▶️  Preprocessing dataset with task preprocessor "
                f"{experiment.task_preprocessor.name}."
            )
            task_preprocessor = experiment.task_preprocessor
            inspect_dataset = task_preprocessor(
                hf_dataset,
                dataset_manager,
                field_spec=experiment.field_spec,
                force_reprocess=force_reload,
            )

            self._load_model(experiment.model_config)

            self._generate_on_dataset(
                experiment,
                inspect_dataset,
                force_rerun=force_rerun,
                eval_kwargs=eval_kwargs,
                eval_retry_kwargs=eval_retry_kwargs,
            )
        self._cleanup_active_model()
        logger.info("✨  Generation tasks completed.")

    def evaluate(
        self,
        show_progress: bool = True,
        force_rerun: bool = False,
        score_kwargs: dict[str, Any] | None = None,
    ):
        """Runs the evaluation stage of the pipeline.

        Args:
            show_progress (bool, optional): Whether to show a progress bar.
                Defaults to True.
            force_rerun (bool, optional): Whether to force rerun the experiments.
                Defaults to False.
            score_kwargs (dict[str, Any], optional): Additional arguments to pass
                to the Inspect score function. Defaults to empty dictionary when
                None.
        """
        experiments_to_evaluate = [
            experiment
            for experiment in self.experiments
            if experiment.evaluator is not None
        ]
        for experiment in tqdm(
            experiments_to_evaluate,
            disable=not show_progress,
            desc="Experiment Evaluation",
        ):
            logger.info(
                f"🔄  Starting evaluation for {experiment.evaluation_record.label}"
            )
            prev_record = self.project.get_record(
                experiment.evaluation_record,
            )
            if prev_record is None or prev_record.log_location is None:
                logger.error("❌  Evaluation skipped — no valid generations found.")
                continue
            if prev_record.status == "success":
                logger.info("⏭️  Evaluation skipped — already completed.")
                continue

            # Prepare the scorer
            evaluator = cast(Evaluator, experiment.evaluator)
            scorer = evaluator.scorer
            if isinstance(scorer, ScorerFactory):
                if evaluator.model_config is None:
                    logger.error(
                        "❌  Using ScorerFactory as a scorer for evaluation requires a "
                        "model config to specify the used model. Skipping evaluation."
                    )
                    continue
                scorer = scorer.create_scorer(self._load_model(evaluator.model_config))

            init_score_log = self.project.get_eval_log(
                experiment.evaluation_record,
            )
            if init_score_log is None:
                logger.error(
                    "❌  Couldn't load initial evaluation log. Skipping evaluation."
                )
                continue

            exception = None
            try:
                score_log = score(
                    log=init_score_log, scorers=scorer, action="overwrite"
                )
            except BaseException as e:
                score_log = self.project.get_eval_log(experiment.evaluation_record)
                exception = e
            score_log = cast(EvalLog, score_log)
            write_eval_log(score_log, location=score_log.location)

            status = "error"
            error_message = "Unknown error"
            log_location = None
            if not score_log:
                error_message = "No log returned from evaluation."
                logger.error("❌  Evaluation failed: no log returned from evaluation.")
            else:
                log_location = score_log.location
                if score_log.status == "error" or exception is not None:
                    if score_log.error is not None:
                        error_message = score_log.error.message
                    elif exception is not None:
                        error_message = str(exception)
                    logger.error(
                        f"❌  Evaluation failed due to an error: {error_message}"
                    )
                elif score_log.status == "cancelled":
                    error_message = "Evaluation was cancelled."
                    logger.error("❌  Evaluation was cancelled.")
                elif score_log.status == "success":
                    status = "success"
                    error_message = None
                    logger.info(
                        f"✅  Evaluation for {experiment.evaluation_record.label} "
                        "completed successfully."
                    )
            self.project.update_record(
                experiment.evaluation_record,
                ResultRecord(
                    status=status,
                    error_message=error_message,
                    log_location=log_location,
                ),
            )

            if isinstance(exception, KeyboardInterrupt):
                logger.critical("🛑  Execution was interrupted.")
                raise KeyboardInterrupt()

        self._cleanup_active_model()
        logger.info("✨  Evaluation tasks completed.")

    def run(
        self,
        show_progress: bool = True,
        force_rerun: bool = False,
        force_reload: bool = False,
        eval_kwargs: dict[str, Any] | None = None,
        eval_retry_kwargs: dict[str, Any] | None = None,
        score_kwargs: dict[str, Any] | None = None,
    ):
        """Runs the pipeline.

        Args:
            show_progress (bool, optional): Whether to show a progress bar.
                Defaults to True.
            force_rerun (bool, optional): Whether to force rerun the experiments.
                Defaults to False.
            force_reload (bool, optional): Whether to force reloading and
                reprocessing the datasets. Defaults to False.
            eval_kwargs (dict[str, Any], optional): Additional arguments to pass
                to the Inspect eval function. Defaults to empty dictionary when
                None.
            eval_retry_kwargs (dict[str, Any], optional): Additional arguments
                to pass to the Inspect eval function for retrying failed tasks.
                Defaults to empty dictionary when None.
            score_kwargs (dict[str, Any], optional): Additional arguments to pass
                to the Inspect score function. Defaults to empty dictionary when
                None.
        """
        self.generate(
            show_progress=show_progress,
            force_rerun=force_rerun,
            force_reload=force_reload,
            eval_kwargs=eval_kwargs,
            eval_retry_kwargs=eval_retry_kwargs,
        )
        self.evaluate(
            show_progress=show_progress,
            force_rerun=force_rerun,
            score_kwargs=score_kwargs,
        )
