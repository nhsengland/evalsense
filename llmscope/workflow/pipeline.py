from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

from llmscope.prompts import PromptFormatter
from llmscope.evaluation import (
    ExperimentConfig,
    ExperimentBatchConfig,
)
from llmscope.utils.huggingface import disable_dataset_progress_bars
from llmscope.workflow.project import Project

type ExperimentDefinitions = (
    ExperimentConfig
    | ExperimentBatchConfig
    | list[ExperimentConfig | ExperimentBatchConfig]
)


class Pipeline[T]:
    """A pipeline for evaluating LLMs."""

    def __init__(
        self,
        experiments: ExperimentDefinitions,
        project: Project[T],
    ):
        """Initializes a new Pipeline.

        Args:
            experiments (ExperimentBatchConfig | ExperimentConfig
                | list[ExperimentBatchConfig | ExperimentConfig]): The experiments
                to run in the pipeline.
            project (Project): The project in which to track the results and outputs.
        """
        # Standardize experiments to a list of ExperimentConfigs
        if not isinstance(experiments, list):
            experiments = [experiments]
        all_experiments: list[ExperimentConfig] = []
        for experiment in experiments:
            if isinstance(experiment, ExperimentBatchConfig):
                experiment.validate()
                all_experiments.extend(experiment.all_experiments())
            else:
                all_experiments.append(experiment)
        self.experiments = all_experiments
        self.project = project

    def _generate_on_dataset(
        self,
        dataset: Dataset,
        experiment: ExperimentConfig,
        split_name: str,
        show_progress: bool = True,
    ) -> Dataset:
        """Generates LLM outputs on a dataset.

        Args:
            dataset (Dataset): The dataset to process.
            experiment (ExperimentConfig): The experiment configuration.
            split_name (str, optional): The name of the split. Defaults to None.
            show_progress (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            Dataset: The dataset including the LLM outputs in the `output` column.
        """

        def map_sample(sample: dict, prompt_formatter: PromptFormatter) -> dict:
            messages = prompt_formatter(**sample)
            sample[experiment.column_config.inputs.column_name] = messages
            return sample

        with disable_dataset_progress_bars():
            dataset = dataset.map(
                map_sample,
                fn_kwargs={"prompt_formatter": experiment.task.prompt_formatter},
                batched=False,
            )

        outputs = experiment.llm_manager(
            dataset[experiment.column_config.inputs.column_name],
            show_progress=show_progress,
        )
        generations_dataset = dataset.add_column(
            experiment.column_config.outputs.column_name, outputs
        )  # type: ignore

        self.project.add_generations(
            generations_dataset,
            experiment.get_id(split_name),
        )

        return generations_dataset

    def _evaluate_on_dataset(
        self,
        generations: Dataset,
        experiment: ExperimentConfig,
        split_name: str,
    ) -> None:
        """Evaluates LLM outputs on a dataset.

        Args:
            generations (Dataset): The dataset with LLM outputs.
            experiment (ExperimentConfig): The experiment configuration.
            split_name (str, optional): The name of the split. Defaults to None.
        """
        if experiment.evaluator is not None:
            eval_results = experiment.evaluator(
                generations, column_config=experiment.column_config
            )
            for result in eval_results:
                self.project.add_result(
                    result=result,
                    experiment_id=experiment.get_id(split_name),
                )

    def run(self, show_progress: bool = True) -> T:
        """Runs the pipeline.

        Args:
            show_progress (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            T: The results of the pipeline.
        """
        for experiment in tqdm(
            self.experiments, disable=not show_progress, desc="Experiments"
        ):
            for split_name in tqdm(
                experiment.task.dataset_manager.splits,
                disable=not show_progress,
                desc="Splits",
            ):
                experiment_id = experiment.get_id(split_name)
                if experiment.evaluator is not None:
                    result_id = experiment_id.to_result_id(experiment.evaluator.name)
                    if self.project.has_result(result_id):
                        # Skip if the result has already been computed
                        continue

                if self.project.has_generations(experiment_id):
                    generations = self.project.get_generations(experiment_id)
                else:
                    dataset = experiment.task.dataset_manager.load()

                    if experiment.task.task_preprocessor is not None:
                        dataset = experiment.task.task_preprocessor(dataset)

                    if isinstance(dataset, DatasetDict):
                        # Only select the currently processed split
                        dataset = dataset[split_name]

                    generations = self._generate_on_dataset(
                        dataset,
                        experiment,
                        split_name=split_name,
                        show_progress=show_progress,
                    )

                self._evaluate_on_dataset(
                    generations,
                    experiment,
                    split_name=split_name,
                )

        return self.project.get_result_summary()
