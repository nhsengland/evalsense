from typing import Literal, cast, override

from datasets import Dataset
import polars as pl

from llmscope.evaluation.result_aggregator import EvaluationResult, ResultAggregator
from llmscope.workflow.experiment import ExperimentId


# TODO: Support returning more detailed results
class DefaultAggregator[T](ResultAggregator[T]):
    """The default aggregator for evaluation results in LLMScope."""

    def __init__(
        self,
        return_instance_results: bool = True,
        return_format: Literal["polars", "pandas", "dataset"] = "polars",
    ) -> None:
        """Initializes the aggregator.

        Args:
            aggregation_fn (AggregationFunction, optional): The aggregation
                function to use. Defaults to mean aggregation.
            return_instance_results (bool, optional): Specifies whether to return
                instance-level results. Defaults to True.
            return_format (str, optional): The type of result to return.
                Can be "polars", "pandas", or "dataset". Defaults to "polars".
        """
        self.return_instance_results = return_instance_results
        self.added_results = set()
        self.overall_results_data = []
        self.return_format = return_format

    @override
    def add_result(
        self,
        *,
        result: EvaluationResult,
        experiment_id: ExperimentId,
        exist_ok: bool = False,
    ) -> None:
        """Adds a result to the aggregator.

        Args:
            result (EvaluationResult): The evaluation result to add.
            experiment_id (ExperimentId): The ID data of the experiment associated
                with the result.
            exist_ok (bool, optional): Whether to allow adding the same result
                multiple times. Defaults to False.
        """
        result_id = experiment_id.to_result_id(metric_name=result.name)
        if result_id in self.added_results and not exist_ok:
            raise ValueError(f"Result with ID {result_id} has already been added.")
        self.added_results.add(result_id)

        self.overall_results_data.append(
            {
                "dataset": experiment_id.dataset_name,
                "split": experiment_id.split_name,
                "task": experiment_id.task_name,
                "prompt": experiment_id.prompt_name,
                "model": experiment_id.model_name,
                "metric": result.name,
                "value": result.overall_result,
            }
        )

    @override
    def return_results(self) -> T:
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
        if self.return_format == "polars":
            return cast(T, df)
        elif self.return_format == "pandas":
            return cast(T, df.to_pandas())
        elif self.return_format == "dataset":
            return cast(T, Dataset.from_polars(df))
        else:
            raise ValueError(f"Unsupported return format: {self.return_format}")
