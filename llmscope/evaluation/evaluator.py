from abc import abstractmethod
from typing import Protocol, overload

from datasets import Dataset, DatasetDict

from llmscope.columns import ColumnConfig
from llmscope.evaluation.evaluation_result import EvaluationResult


class Evaluator(Protocol):
    """A protocol for LLM output evaluators.

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

    # TODO: Move to ColumnConfig?
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

    @overload
    def __call__(
        self,
        dataset: Dataset,
        column_config: ColumnConfig = ColumnConfig(),
        **kwargs,
    ) -> list[EvaluationResult]: ...

    @overload
    def __call__(
        self,
        dataset: DatasetDict,
        column_config: ColumnConfig = ColumnConfig(),
        **kwargs,
    ) -> dict[str, list[EvaluationResult]]: ...

    def __call__(
        self,
        dataset: Dataset | DatasetDict,
        column_config: ColumnConfig | None = None,
        **kwargs,
    ) -> list[EvaluationResult] | dict[str, list[EvaluationResult]]:
        """Evaluates the LLM outputs on a dataset.

        Args:
            dataset (Dataset | DatasetDict): The dataset containing the data
                and LLM outputs to evaluate.
            output_column (str): The column containing the LLM outputs.
            column_config (ColumnsConfig, optional): The column configuration to
                use during evaluation. Specifies the mapping between the dataset
                columns and the parameters of the evaluation function.
                Defaults to the built-in `ColumnConfig` from `llmscope.columns`.
            **kwargs: Additional keyword arguments.

        Returns:
            (list[EvaluationResult] | dict[str, list[EvaluationResult]]):
                The evaluation results. If a single Dataset is provided,
                a single list[EvaluationResult] is returned. Otherwise,
                a dict of list[EvaluationResult] is returned, with the keys
                corresponding to the dataset splits.
        """
        if column_config is None:
            column_config = ColumnConfig()

        if isinstance(dataset, DatasetDict):
            results = {}
            for split, ds in dataset.items():
                eval_columns = self._extract_columns(ds, column_config)
                results[split] = self._evaluate_outputs(**eval_columns)
        else:
            eval_columns = self._extract_columns(dataset, column_config)
            results = self._evaluate_outputs(**eval_columns, **kwargs)

        return results
