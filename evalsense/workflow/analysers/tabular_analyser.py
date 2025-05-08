from typing import Literal, cast, override

import pandas as pd
import polars as pl

from evalsense.workflow import Project, ResultAnalyser

OUTPUT_FORMATTERS = {
    "polars": lambda df: df,
    "pandas": lambda df: df.to_pandas(),
}


class TabularResultAnalyser[T: pl.DataFrame | pd.DataFrame](ResultAnalyser[T]):
    """A class for summarising evaluation results in a tabular format.

    This class is generic in T to provide better type hints when returning
    different output types. It is the responsibility of the client code to
    ensure that the specified `output_format` is compatible with the type T.
    For example, a correct use of this class could look as follows:

        analyser = TabularResultAnalyser[pl.DataFrame](
            output_format="polars",
        )
    """

    def __init__(
        self,
        name: str = "TabularResultAnalyser",
        output_format: Literal["polars", "pandas"] = "polars",
    ):
        """Initializes the tabular result analyser.

        Args:
            name (str): The name of the tabular result analyser.
            output_format (Literal["polars", "pandas", "dataset"]): The output format of the
                result. Can be "polars" or "pandas". Defaults to "polars".
        """
        super().__init__(name=name)
        if output_format not in OUTPUT_FORMATTERS:
            raise ValueError(
                f"Invalid output format: {output_format}. "
                f"Must be one of: {', '.join(OUTPUT_FORMATTERS.keys())}."
            )
        self.output_format = output_format

    @override
    def __call__(self, project: Project, **kwargs: dict) -> T:
        """Analyses the evaluation results.

        Args:
            project (Project): The project holding the evaluation data to analyse.
            **kwargs (dict): Additional arguments for the analysis.

        Returns:
            T: The analysed results in the specified output format.
        """
        eval_logs = project.get_logs(type="evaluation", status="success")

        result_data = []
        for eval_record, log in eval_logs.items():
            if not log.results:
                continue

            for score in log.results.scores:
                for metric_name, metric in score.metrics.items():
                    value = metric.value

                    result_data.append(
                        {
                            "dataset": eval_record.dataset_record.name,
                            "splits": ", ".join(eval_record.dataset_record.splits),
                            "task": eval_record.task_name,
                            "generator": eval_record.generator_name,
                            "model": eval_record.model_record.name,
                            "metric": f"{score.name}/{metric_name}",
                            "value": value,
                        }
                    )

        df = pl.DataFrame(result_data)
        df = df.pivot(
            on="metric",
            index=["dataset", "splits", "task", "generator", "model"],
            values="value",
            aggregate_function="first",
        )
        if self.output_format in OUTPUT_FORMATTERS:
            return cast(T, OUTPUT_FORMATTERS[self.output_format](df))
        raise ValueError(
            f"Invalid output format: {self.output_format}. "
            f"Must be one of: {', '.join(OUTPUT_FORMATTERS.keys())}."
        )
