from abc import abstractmethod
from typing import Protocol

from datasets import Dataset, DatasetDict, concatenate_datasets

from llmscope.columns import ColumnConfig
from llmscope.evaluation.evaluation_result import EvaluationResult


class Evaluator(Protocol):
    """A protocol defining the interface for LLM output evaluators.

    Attributes:
        name (str): The name of the evaluator.
    """

    name: str

    def __init__(self, name: str):
        """Initializes the evaluator with a name.

        Args:
            name (str): The name of the evaluator.
        """
        self.name = name

    @abstractmethod
    def _evaluate_outputs(
        self,
        outputs: list[str],
        references: list[str] | list[list[str]] | None = None,
        inputs: list[str] | None = None,
        contexts: list[str] | None = None,
        retrieval_contexts: list[str] | None = None,
        **kwargs,
    ) -> list[EvaluationResult]:
        """Evaluates the LLM outputs. Called by the main `evaluate` method.

        Args:
            outputs (list[str]): The LLM outputs to evaluate.
            references (list[str] | list[list[str]] | None): The reference values,
                if available.
            inputs (list[str] | None): The input values, if available.
            contexts (list[str] | None): The context values, if available.
            retrieval_contexts (list[str] | None): The retrieval context values,
                if available.
            **kwargs: Additional keyword arguments.

        Returns:
            list[EvaluationResult]: The evaluation results.
        """
        ...

    def _extract_columns(
        self,
        dataset: Dataset,
        column_config: ColumnConfig,
    ) -> dict[str, list]:
        """Extracts the evaluation-related columns from the dataset.

        Args:
            dataset (Dataset): The dataset to extract columns from.
            column_config (ColumnConfig): The column configuration to use.

        Returns:
            (dict[str, list]): A dictionary containing the extracted columns.
        """
        extracted_columns = {}
        for base_name, column in column_config.all_columns.items():
            if (
                column.required_for_eval
                and column.column_name not in dataset.column_names
            ):
                raise ValueError(
                    f"Column '{column.column_name}' is required for evaluation but not found in the dataset."
                )
            if column.column_name in dataset.column_names:
                extracted_columns[base_name] = dataset[column.column_name]
        return extracted_columns

    def evaluate(
        self,
        dataset: Dataset | DatasetDict,
        column_config: ColumnConfig = ColumnConfig(),
        merge_splits: bool = True,
        **kwargs,
    ) -> list[EvaluationResult] | dict[str, list[EvaluationResult]]:
        """Evaluates the LLM outputs on a dataset.

        Args:
            dataset (Dataset | DatasetDict): The dataset to evaluate.
            output_column (str): The column containing the LLM outputs.
            column_config (ColumnsConfig, optional): The column configuration to
                use during evaluation. Defaults to the default `ColumnConfig`
                from `llmscope.columns`.
            merge_splits (bool): Specifies whether to merge the dataset splits into
                a single dataset.
            **kwargs: Additional keyword arguments.

        Returns:
            (list[EvaluationResult] | dict[str, list[EvaluationResult]]):
                The evaluation results. If a single Dataset is provided or
                `merge_splits` is True, a single list[EvaluationResult] is returned.
                Otherwise, a dict of list[EvaluationResult] is returned, with the keys
                corresponding to the dataset splits.
        """
        if merge_splits and isinstance(dataset, DatasetDict):
            dataset = concatenate_datasets(list(dataset.values()))

        if isinstance(dataset, DatasetDict):
            results = {}
            for split, ds in dataset.items():
                eval_columns = self._extract_columns(ds, column_config)
                results[split] = self._evaluate_outputs(**eval_columns)
        else:
            eval_columns = self._extract_columns(dataset, column_config)
            results = self._evaluate_outputs(**eval_columns, **kwargs)

        return results
