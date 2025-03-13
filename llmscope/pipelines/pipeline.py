from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

import polars as pl

from llmscope.columns import ColumnConfig
from llmscope.datasets import DatasetManager
from llmscope.evaluation import Evaluator
from llmscope.evaluation.aggregators import DefaultAggregator
from llmscope.llms import LlmManager
from llmscope.prompts import PromptFormatter
from llmscope.tasks import TaskPreprocessor
from llmscope.utils.huggingface import disable_dataset_progress_bars


class SimplePipeline:
    """A simple pipeline for processing data using an LLM.

    Attributes:
        dataset_manager (DatasetManager): The dataset manager.
        task_preprocessor (TaskPreprocessor): The task preprocessor.
        prompt_formatter (PromptFormatter | None): The prompt formatter.
        llm_manager (LlmManager): The LLM manager.
        column_config (ColumnConfig): The column configuration.
    """

    def __init__(
        self,
        *,
        dataset_manager: DatasetManager,
        task_preprocessor: TaskPreprocessor | None = None,
        prompt_formatter: PromptFormatter,
        llm_manager: LlmManager,
        evaluator: Evaluator,
        column_config: ColumnConfig | None = None,
    ):
        """Initializes a new SimplePipeline.

        Args:
            dataset_manager (DatasetManager): The dataset manager.
            task_preprocessor (TaskPreprocessor, optional): The task preprocessor.
            prompt_formatter (PromptFormatter): The prompt formatter.
            llm_manager (LlmManager): The LLM manager.
            evaluator (Evaluator): The evaluator.
            column_config (ColumnsConfig, optional): The column configuration.
                Defaults to the default `ColumnConfig` from `llmscope.columns`.
        """
        if column_config is None:
            column_config = ColumnConfig()
        self.dataset_manager = dataset_manager
        self.task_preprocessor = task_preprocessor
        self.prompt_formatter = prompt_formatter
        self.llm_manager = llm_manager
        self.evaluator = evaluator
        self.column_config = column_config
        # TODO: Prototype code — hardcoded aggregator
        self.result_aggregator = DefaultAggregator()

    def _run_dataset(self, dataset: Dataset, show_progress=True) -> Dataset:
        """Runs the pipeline on a dataset.

        Args:
            dataset (Dataset): The dataset to process.

        Returns:
            (Dataset): The dataset including the LLM outputs in the `output` column.
        """

        def map_sample(sample: dict, prompt_formatter: PromptFormatter) -> dict:
            messages = prompt_formatter(**sample)
            sample[self.column_config.inputs.column_name] = messages
            return sample

        with disable_dataset_progress_bars():
            dataset = dataset.map(
                map_sample,
                fn_kwargs={"prompt_formatter": self.prompt_formatter},
                batched=False,
            )

        outputs = self.llm_manager(
            dataset[self.column_config.inputs.column_name],
            show_progress=show_progress,
        )
        dataset = dataset.add_column(self.column_config.outputs.column_name, outputs)  # type: ignore
        return dataset

    def run(self, show_progress=True) -> pl.DataFrame:
        """Runs the pipeline.

        Args:
            show_progress (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            (T): The dataset including the LLM outputs in the `output` column.
        """
        # Load the dataset
        dataset = self.dataset_manager()

        # Preprocess the dataset
        if self.task_preprocessor:
            dataset = self.task_preprocessor(dataset)

        # Produce LLM outputs
        if isinstance(dataset, DatasetDict):
            output_dict = {}
            for split_name, dataset in (
                progress := tqdm(dataset.items(), disable=not show_progress)
            ):
                progress.set_description(f"Running on split {split_name}")
                output_dict[split_name] = self._run_dataset(dataset, show_progress)
            outputs = DatasetDict(output_dict)
        else:
            outputs = self._run_dataset(dataset, show_progress)

        # TODO: Prototype code here — unhandled cases and issues with typing
        if isinstance(outputs, DatasetDict):
            raise NotImplementedError(
                "Aggregating results for multiple splits is not yet supported."
            )

        evaluation_results = self.evaluator.evaluate(outputs, self.column_config)
        if not isinstance(evaluation_results, list):
            raise NotImplementedError(
                "Aggregating results for multiple splits is not yet supported."
            )
        for result in evaluation_results:
            self.result_aggregator.add_result(
                result=result,
                dataset=outputs,
                dataset_name=self.dataset_manager.name,
                task_name="generate note",
                prompt_name="MEDIC prompt",
                model_name=self.llm_manager.name,
            )

        return self.result_aggregator.return_results(return_format="polars")
