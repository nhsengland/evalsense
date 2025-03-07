from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

from llmscope.constants import INPUT_COLUMN, OUTPUT_COLUMN
from llmscope.datasets import DatasetManager
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
    """

    def __init__(
        self,
        *,
        dataset_manager: DatasetManager,
        task_preprocessor: TaskPreprocessor | None = None,
        prompt_formatter: PromptFormatter,
        llm_manager: LlmManager,
    ):
        """Initializes a new SimplePipeline.

        Args:
            dataset_manager (DatasetManager): The dataset manager.
            task_preprocessor (TaskPreprocessor, optional): The task preprocessor.
            prompt_formatter (PromptFormatter): The prompt formatter.
            llm_manager (LlmManager): The LLM manager.
        """
        self.dataset_manager = dataset_manager
        self.task_preprocessor = task_preprocessor
        self.prompt_formatter = prompt_formatter
        self.llm_manager = llm_manager

    def _run_dataset(self, dataset: Dataset, show_progress=True) -> Dataset:
        """Runs the pipeline on a dataset.

        Args:
            dataset (Dataset): The dataset to process.

        Returns:
            (Dataset): The dataset including the LLM outputs in the `output` column.
        """

        def map_sample(sample: dict, prompt_formatter: PromptFormatter) -> dict:
            messages = prompt_formatter(**sample)
            sample[INPUT_COLUMN] = messages
            return sample

        with disable_dataset_progress_bars():
            dataset = dataset.map(
                map_sample,
                fn_kwargs={"prompt_formatter": self.prompt_formatter},
                batched=False,
            )

        outputs = self.llm_manager(
            dataset[INPUT_COLUMN],
            show_progress=show_progress,
        )
        dataset = dataset.add_column(OUTPUT_COLUMN, outputs)  # type: ignore
        return dataset

    def run(self, show_progress=True) -> Dataset | DatasetDict:
        """Runs the pipeline.

        Args:
            show_progress (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            (Dataset | DatasetDict): The dataset including the LLM outputs in the `output` column.
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

        return outputs
