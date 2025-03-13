from typing import Literal, overload, override

from datasets import Dataset
import pandas as pd
import polars as pl

from llmscope.evaluation import EvaluationResult, ResultAggregator


# TODO: Support returning more detailed results
class DefaultAggregator(ResultAggregator[pl.DataFrame | pd.DataFrame | Dataset]):
    """The default aggregator for evaluation results in LLMScope."""

    def __init__(
        self,
        return_instance_results: bool = True,
    ) -> None:
        """Initializes the aggregator.

        Args:
            aggregation_fn (AggregationFunction, optional): The aggregation
                function to use. Defaults to mean aggregation.
            return_instance_results (bool, optional): Specifies whether to return
                instance-level results. Defaults to True.
        """
        self.return_instance_results = return_instance_results
        self.overall_results_data = []

    @override
    def add_result(
        self,
        *,
        result: EvaluationResult,
        dataset: Dataset,
        dataset_name: str,
        task_name: str,
        prompt_name: str,
        model_name: str,
    ) -> None:
        """Adds a result to the aggregator.

        Args:
            result (EvaluationResult): The evaluation result to add.
            dataset (Dataset): The dataset associated with the result.
            dataset_name (str): The name of the dataset.
            task_name (str): The name of the task.
            prompt_name (str): The name of the prompt.
            model_name (str): The name of the model.
        """
        self.overall_results_data.append(
            {
                "dataset": dataset_name,
                "task": task_name,
                "prompt": prompt_name,
                "model": model_name,
                "metric": result.name,
                "value": result.overall_result,
            }
        )

    @overload
    def return_results(self, return_format: Literal["polars"]) -> pl.DataFrame: ...

    @overload
    def return_results(self, return_format: Literal["pandas"]) -> pd.DataFrame: ...

    @overload
    def return_results(self, return_format: Literal["dataset"]) -> Dataset: ...

    @override
    def return_results(
        self, return_format: Literal["polars", "pandas", "dataset"] = "polars"
    ) -> pl.DataFrame | pd.DataFrame | Dataset:
        """Aggregates the results.

        Returns:
            (pl.DataFrame | pd.DataFrame | Dataset): The aggregated result.
        """
        df = pl.DataFrame(self.overall_results_data)
        df = df.pivot(
            on="metric",
            index=["dataset", "task", "prompt", "model"],
            values="value",
            aggregate_function="first",
        )
        if return_format == "polars":
            return df
        elif return_format == "pandas":
            return df.to_pandas()
        elif return_format == "dataset":
            return Dataset.from_polars(df)
        else:
            raise ValueError(f"Unsupported return format: {return_format}")
