from collections import defaultdict
import math
from typing import Any, Literal, cast, override

import pandas as pd
import polars as pl
from scipy.stats import spearmanr

from evalsense.evaluation import MetaTierGroupedRecord
from evalsense.workflow import Project, ResultAnalyser

OUTPUT_FORMATTERS = {
    "polars": lambda df: df,
    "pandas": lambda df: df.to_pandas(),
}


class MetaResultAnalyser[T: pl.DataFrame | pd.DataFrame](ResultAnalyser[T]):
    """An analyser for conducing a meta-evaluation of different evaluation methods.

    The analyser computes the Spearman rank correlation between the rankings specified by
    the meta tiers and the scores returned by the evaluation methods.
    The meta tiers can either be sourced from human annotations or be based on
    progressive perturbations for automatic meta-evaluation.
    """

    def __init__(
        self,
        name: str = "MetaResultAnalyser",
        output_format: Literal["polars", "pandas", "numpy"] = "polars",
    ):
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
        meta_tier_field: str = "perturbation_type_tier",
        lower_tier_is_better: bool = False,
        metric_labels: dict[str, str] | None = None,
        **kwargs: dict[str, Any],
    ) -> T:
        """
        Analyses the results from perturbation-based meta-evaluation experiments.

        Args:
            project (Project): The project holding the meta-evaluation data to analyse.
            meta_tier_field (str): The field name that indicates the meta-evaluation
                tier to specify the expected score ranking.
            lower_tier_is_better (bool): If True, lower perturbation tiers correspond
                to better outputs. If False, higher tiers are better. Defaults to False.
            metric_labels (dict[str, str] | None): A dictionary mapping metric names
                to their labels in the output table. If None, no aliasing is performed.
                Defaults to None.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            T: The analysed results in the specified output format.
        """
        eval_logs = project.get_logs(type="evaluation", status="success")

        # Data structure for tracking the intermediate results
        # The nested dictionary is indexed by perturbation record → sample ID → perturbation tier
        result_data: dict[
            MetaTierGroupedRecord, dict[str | int, dict[int, float | int]]
        ] = defaultdict(lambda: defaultdict(dict))

        for eval_record, log in eval_logs.items():
            if not hasattr(log, "samples") or not log.samples:
                continue

            # Extract scores for the individual samples
            for sample in log.samples:
                if not hasattr(sample, "scores") or not sample.scores:
                    continue

                if meta_tier_field not in sample.metadata:
                    raise ValueError(
                        f"Meta tier field '{meta_tier_field}' not found in sample metadata."
                    )
                meta_tier = int(cast(int, sample.metadata.get(meta_tier_field)))
                sample_id = sample.id

                for metric_name, score in sample.scores.items():
                    if type(score.value) is float or type(score.value) is int:
                        if metric_labels is not None and metric_name in metric_labels:
                            metric_name = metric_labels[metric_name]

                        result_data[eval_record.get_meta_grouped_record(metric_name)][
                            sample_id
                        ][meta_tier] = score.value
                    elif type(score.value) is dict:
                        # Extract inner scores from result dictionary
                        for inner_metric_name, inner_score in score.value.items():
                            if (
                                metric_labels is not None
                                and inner_metric_name in metric_labels
                            ):
                                inner_metric_name = metric_labels[inner_metric_name]

                            if type(inner_score) is float or type(inner_score) is int:
                                result_data[
                                    eval_record.get_meta_grouped_record(
                                        inner_metric_name
                                    )
                                ][sample_id][meta_tier] = inner_score
            del log

        # For each metric, compute average spearman rank correlation between the
        # meta tiers and the scores
        correlation_data: dict[str, list[float]] = defaultdict(list)
        for perturbation_record, samples in result_data.items():
            for sample_id, perturbation_scores in samples.items():
                perturbation_tiers = list(perturbation_scores.keys())
                perturbation_values = list(perturbation_scores.values())
                multiplier = -1 if lower_tier_is_better else 1
                correlation = spearmanr(
                    [(multiplier * pt) for pt in perturbation_tiers],
                    perturbation_values,
                ).correlation  # type: ignore
                if math.isnan(correlation):
                    continue
                correlation_data[perturbation_record.metric_name].append(correlation)

        correlation_results = []
        for metric_name, correlations in correlation_data.items():
            correlation_results.append(
                {
                    "metric_name": metric_name,
                    "avg_correlation": sum(correlations) / len(correlations),
                }
            )

        df = pl.DataFrame(correlation_results)
        if self.output_format in OUTPUT_FORMATTERS:
            return cast(T, OUTPUT_FORMATTERS[self.output_format](df))
        raise ValueError(
            f"Invalid output format: {self.output_format}. "
            f"Must be one of: {', '.join(OUTPUT_FORMATTERS.keys())}."
        )
