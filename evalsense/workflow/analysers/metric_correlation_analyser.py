from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Literal, cast, override

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from evalsense.workflow import Project, ResultAnalyser

OUTPUT_FORMATTERS = {
    "polars": lambda df: df,
    "pandas": lambda df: df.to_pandas(),
    "numpy": lambda df: df.to_numpy(),
}


@dataclass
class CorrelationResults[T: pl.DataFrame | pd.DataFrame | npt.NDArray[np.float_]]:
    """Class to hold correlation analysis results."""

    correlation_matrix: T
    figure: Figure | None = None


class MetricCorrelationAnalyser[T: CorrelationResults](ResultAnalyser[T]):
    """A class for calculating and visualizing Spearman rank correlations between
    different evaluation metrics.

    This class analyzes the correlation between scores returned for individual samples
    by pairs of different evaluation methods, and produces a correlation matrix plot.
    """

    def __init__(
        self,
        name: str = "MetricCorrelationAnalyser",
        output_format: Literal["polars", "pandas", "numpy"] = "polars",
    ):
        """Initializes the metric correlation analyser.

        Args:
            name (str): The name of the metric correlation analyser.
            output_format (Literal["polars", "pandas", "numpy"]): The output
                format of the correlation matrix. Can be "polars", "pandas",
                or "numpy". Defaults to "polars".
        """
        super().__init__(name=name)
        if output_format not in OUTPUT_FORMATTERS:
            raise ValueError(
                f"Invalid output format: {output_format}. "
                f"Must be one of: {', '.join(OUTPUT_FORMATTERS.keys())}."
            )
        self.output_format = output_format

    @override
    def __call__(
        self,
        project: Project,
        corr_method: Literal["spearman", "pearson"] = "spearman",
        return_plot: bool = True,
        figsize: tuple[int, int] = (12, 10),
        metric_labels: dict[str, str] | None = None,
        method_filter_fun: Callable[[str], bool] = lambda _: True,
        **kwargs: dict,
    ) -> T:
        """Calculates Spearman rank correlations between evaluation metrics.

        Args:
            project (Project): The project holding the evaluation data to analyse.
            corr_method (Literal["spearman", "pearson"]): The correlation method to use.
                Can be "spearman" or "pearson". Defaults to "spearman".
            return_plot (bool): Whether to generate and return a visualization of the
                correlation matrix. Defaults to True.
            figsize (Tuple[int, int]): Figure size for the correlation matrix plot.
                Defaults to (10, 8).
            metric_labels (dict[str, str] | None): A dictionary mapping metric names
                to their labels in the figure. If None, no aliasing is performed.
                Defaults to None.
            **kwargs (dict): Additional arguments for the analysis.

        Returns:
            T: The correlation results containing the correlation matrix and
                optionally a visualization.
        """
        eval_logs = project.get_logs(type="evaluation", status="success")

        result_data: dict[str, list[float | int]] = defaultdict(list)
        for log in eval_logs.values():
            if not hasattr(log, "samples") or not log.samples:
                continue

            # Extract scores from individual samples
            for sample in log.samples:
                if not hasattr(sample, "scores") or not sample.scores:
                    continue

                for metric_name, score in sample.scores.items():
                    if not method_filter_fun(metric_name):
                        continue

                    if metric_labels is not None and metric_name in metric_labels:
                        metric_name = metric_labels[metric_name]

                    if type(score.value) is float or type(score.value) is int:
                        result_data[metric_name].append(score.value)
                    elif type(score.value) is dict:
                        # Extract inner scores from result dictionary
                        for inner_metric_name, inner_score in score.value.items():
                            if not method_filter_fun(inner_metric_name):
                                continue

                            if (
                                metric_labels is not None
                                and inner_metric_name in metric_labels
                            ):
                                inner_metric_name = metric_labels[inner_metric_name]

                            if type(inner_score) is float or type(inner_score) is int:
                                result_data[inner_metric_name].append(inner_score)

        sample_scores_df = pl.DataFrame(result_data)

        correlation_data = sample_scores_df.select(
            pl.corr(
                sample_scores_df.get_column(col1),
                sample_scores_df.get_column(col2),
                method=corr_method,
            ).alias(f"{col1}__{col2}")
            for i, col1 in enumerate(sample_scores_df.columns)
            for col2 in sample_scores_df.columns[i:]
        )

        # Reshape the correlation data to a proper matrix format
        cols = sample_scores_df.columns
        matrix_data = [[0.0 for _ in cols] for _ in cols]

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i <= j:
                    col_name = f"{col1}__{col2}"
                    if col_name in correlation_data.columns:
                        val = correlation_data.get_column(col_name)[0]
                        matrix_data[i][j] = val
                        matrix_data[j][i] = val  # Matrix is symmetric

        # Create the correlation matrix
        corr_matrix = pl.DataFrame(
            matrix_data,
            schema=cols,
        )
        # Add metric names as a first column
        corr_matrix = corr_matrix.with_columns(
            pl.Series(name="metric", values=cols)
        ).select("metric", *cols)

        # Create a visualization of the correlation matrix if requested
        fig = None
        if return_plot:
            # Convert to pandas for visualization with seaborn
            corr_matrix_pd = corr_matrix.to_pandas().set_index("metric")

            fig, ax = plt.subplots(figsize=figsize)
            mask = np.triu(np.ones_like(corr_matrix_pd, dtype=bool), k=1)
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            sns.heatmap(
                corr_matrix_pd,
                mask=mask,
                cmap=cmap,
                vmax=1.0,
                vmin=-1.0,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.5},
                annot=True,
                fmt=".2f",
                ax=ax,
            )

            plt.title("Spearman Rank Correlation Between Evaluation Metrics")
            plt.tight_layout()

        # Format the output according to the specified format
        if self.output_format in OUTPUT_FORMATTERS:
            formatted_corr_matrix = OUTPUT_FORMATTERS[self.output_format](corr_matrix)
            return cast(
                T,
                CorrelationResults(
                    correlation_matrix=formatted_corr_matrix, figure=fig
                ),
            )

        raise ValueError(
            f"Invalid output format: {self.output_format}. "
            f"Must be one of: {', '.join(OUTPUT_FORMATTERS.keys())}."
        )
