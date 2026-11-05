from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from typing import Callable, Literal, cast, override

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from inspect_ai.log import EvalLog

from evalsense.evaluation import GenerationRecord
from evalsense.workflow import Project, ResultAnalyser

OUTPUT_FORMATTERS = {
    "polars": lambda df: df,
    "pandas": lambda df: df.to_pandas(),
    "numpy": lambda df: df.to_numpy(),
}

type ObservationKey = tuple[GenerationRecord, tuple[str, ...], type, str | int, int]
type MetricKey = tuple[str, str, str | None]


@dataclass
class CorrelationResults[T: pl.DataFrame | pd.DataFrame | npt.NDArray[np.float64]]:
    """Correlation matrices and counts, with the same metric order in each table.

    Attributes:
        correlation_matrix (T): Pairwise coefficients in the requested output format.
        figure (Figure | None): Correlation plot, or None when no plot was produced.
        paired_counts (T | None): Number of finite, identity-matched observations
            used for each coefficient. Populated by MetricCorrelationAnalyser.
        excluded_counts (T | None): Number of identity-matched pairs excluded
            because either score is non-finite. Unmatched observations are not
            included in either count. Populated by MetricCorrelationAnalyser.
    """

    correlation_matrix: T
    figure: Figure | None = None
    paired_counts: T | None = None
    excluded_counts: T | None = None


def _source_identity(log: EvalLog) -> tuple[str, ...]:
    """Use the source evaluation ID, or legacy run/task IDs when unavailable."""
    eval_id = getattr(log.eval, "eval_id", None)
    if eval_id is not None:
        if isinstance(eval_id, str) and eval_id:
            return ("eval", eval_id)
    else:
        # Inspect releases before eval_id identify a source task within its run.
        # Both IDs are copied with the generation log into evaluation logs.
        run_id = log.eval.run_id
        task_id = log.eval.task_id
        if isinstance(run_id, str) and run_id and isinstance(task_id, str) and task_id:
            return ("legacy", run_id, task_id)
    raise ValueError(
        "Cannot pair observations without a source eval ID or legacy run/task IDs."
    )


def _metric_names(
    keys: list[MetricKey],
    evaluator_names: dict[MetricKey, str],
    aliases: dict[str, str],
) -> list[str]:
    """Keep inner names unless multiple scorers use the same display name."""
    names = [key[2] if key[2] is not None else key[1] for key in keys]
    counts = Counter(names)
    labels = []
    for key, name in zip(keys, names):
        label = aliases.get(name, name)
        if counts[name] > 1:
            parts = list(dict.fromkeys([evaluator_names[key], key[1], label]))
            label = " / ".join(parts)
        labels.append(label)
    # Display aliases must never merge independent score vectors.
    if len(set(labels)) != len(labels) or "Metric" in labels:
        raise ValueError(
            "Metric labels are ambiguous or use the reserved name 'Metric'. "
            "This can also occur when earlier evaluator configurations remain "
            "in the project under the same names. Use distinct evaluator/scorer "
            "names and metric labels, or analyse configurations separately."
        )
    return labels


def _matrix(values: list[list], names: list[str], dtype: pl.DataType) -> pl.DataFrame:
    return pl.DataFrame(
        {"Metric": pl.Series(names, dtype=pl.String)}
        | {
            name: pl.Series([row[j] for row in values], dtype=dtype)
            for j, name in enumerate(names)
        }
    )


class MetricCorrelationAnalyser[T: CorrelationResults](ResultAnalyser[T]):
    """Correlate scores on the same generated observations.

    Pairing uses the full generation record, source Inspect ``eval_id``, sample ID
    (including its type), and epoch. Older logs without ``eval_id`` require both
    ``run_id`` and ``task_id``. Copies of the same generation can be compared across
    evaluators; independent generation attempts are not paired by sample ID alone.

    Evaluator, scorer and inner-metric identities remain separate from display
    labels. Names are qualified when needed; ambiguous labels raise ValueError
    rather than merge distinct metrics. Retained evaluator configurations can
    share display names even when their identities differ.

    Non-finite scores are excluded pairwise and counted. Fewer than two finite
    pairs or a constant vector produces NaN, including on the diagonal. Pairwise
    exclusion does not establish unbiased missingness; these are descriptive
    correlations without confidence estimates.

    All tables use the same metric order and requested output format. The first
    column, ``Metric``, contains row labels, including in NumPy output. Empty
    selections return empty tables and no figure. See CorrelationResults for the
    definitions of ``paired_counts`` and ``excluded_counts``.
    """

    def __init__(
        self,
        name: str = "MetricCorrelationAnalyser",
        output_format: Literal["polars", "pandas", "numpy"] = "polars",
    ):
        """Initialise the metric correlation analyser.

        Args:
            name (str): Analyser name. Defaults to "MetricCorrelationAnalyser".
            output_format (Literal["polars", "pandas", "numpy"]): Format of all
                result tables. Defaults to "polars".

        Raises:
            ValueError: If output_format is unsupported.
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
        """Calculate pairwise correlations after joining observation identities.

        Args:
            project (Project): Project containing the successful evaluation logs
                to analyse.
            corr_method (Literal["spearman", "pearson"]): Correlation method.
                Defaults to "spearman".
            return_plot (bool): Whether to include a correlation plot when metrics
                are selected. Defaults to True.
            figsize (tuple[int, int]): Plot dimensions. Defaults to (12, 10).
            metric_labels (dict[str, str] | None): Map original scalar scorer names
                or inner metric names to display aliases in all tables and the
                plot. Aliases do not change observation identity. Defaults to None.
            method_filter_fun (Callable[[str], bool]): Include a numeric metric
                when this function returns True for its original scalar scorer
                name or inner metric name, before aliasing. Defaults to including
                all numeric metrics.
            **kwargs (dict): Additional analyser-interface arguments. Unused.

        Returns:
            T: CorrelationResults containing correlation_matrix, paired_counts and
                excluded_counts in the requested output format, and an optional
                figure. Empty selections produce empty tables and no figure.

        Raises:
            ValueError: If corr_method is unsupported, source identity is missing,
                a metric/observation key is duplicated (even with equal scores),
                or display labels are ambiguous or use the reserved name "Metric".
        """
        if corr_method not in ("spearman", "pearson"):
            raise ValueError("corr_method must be 'spearman' or 'pearson'.")
        observations: dict[MetricKey, dict[ObservationKey, float]] = defaultdict(dict)
        evaluator_names: dict[MetricKey, str] = {}
        for record, log in project.get_logs(
            type="evaluation", status="success"
        ).items():
            if not log.samples:
                continue
            generation = record.generation_record
            # Evaluation logs are copies of a generation log, so eval_id identifies
            # their common source. Paths and scoring timestamps do not.
            source_id = _source_identity(log)
            generation_fields = generation.model_dump()
            evaluator_identity = json.dumps(
                {
                    key: value
                    for key, value in record.model_dump(mode="json").items()
                    if key not in generation_fields
                },
                sort_keys=True,
            )
            for sample in log.samples:
                if not sample.scores:
                    continue
                observation = (
                    generation,
                    source_id,
                    type(sample.id),
                    sample.id,
                    sample.epoch,
                )
                for scorer_name, score in sample.scores.items():
                    components = (
                        score.value.items()
                        if isinstance(score.value, dict)
                        else [(None, score.value)]
                    )
                    for inner_name, value in components:
                        name = inner_name if inner_name is not None else scorer_name
                        if type(value) not in (int, float) or not method_filter_fun(
                            name
                        ):
                            continue
                        key = (evaluator_identity, scorer_name, inner_name)
                        if observation in observations[key]:
                            raise ValueError(
                                f"Duplicate observation for {record.evaluator_name!r}, "
                                f"scorer {scorer_name!r}, metric {name!r}, "
                                f"sample {sample.id!r}, epoch {sample.epoch}, "
                                f"source eval {source_id!r}."
                            )
                        observations[key][observation] = float(cast(float | int, value))
                        evaluator_names[key] = record.evaluator_name

        keys = sorted(observations, key=lambda key: (key[0], key[1], key[2] or ""))
        names = _metric_names(keys, evaluator_names, metric_labels or {})
        size = len(keys)
        correlations = [[math.nan] * size for _ in keys]
        paired = [[0] * size for _ in keys]
        excluded = [[0] * size for _ in keys]
        for i, left in enumerate(keys):
            for j in range(i, size):
                right = keys[j]
                # Join before filtering values; never align by vector position.
                common = observations[left].keys() & observations[right].keys()
                pairs = sorted(
                    (observations[left][obs], observations[right][obs])
                    for obs in common
                    if math.isfinite(observations[left][obs])
                    and math.isfinite(observations[right][obs])
                )
                paired[i][j] = paired[j][i] = len(pairs)
                excluded[i][j] = excluded[j][i] = len(common) - len(pairs)
                if len(pairs) < 2:
                    continue
                x, y = zip(*pairs)
                if len(set(x)) < 2 or len(set(y)) < 2:
                    continue
                value = (
                    pl.DataFrame(
                        {
                            "left": pl.Series(x, dtype=pl.Float64),
                            "right": pl.Series(y, dtype=pl.Float64),
                        }
                    )
                    .select(
                        pl.corr(
                            "left", "right", method=corr_method, propagate_nans=True
                        )
                    )
                    .item()
                )
                correlations[i][j] = correlations[j][i] = (
                    float(value) if value is not None else math.nan
                )

        corr_matrix = _matrix(correlations, names, pl.Float64())
        count_matrix = _matrix(paired, names, pl.Int64())
        excluded_matrix = _matrix(excluded, names, pl.Int64())
        fig = None
        if return_plot and names:
            corr_matrix_pd = corr_matrix.to_pandas().set_index("Metric")
            fig, ax = plt.subplots(figsize=figsize)
            mask = np.triu(np.ones_like(corr_matrix_pd, dtype=bool), k=1)
            sns.heatmap(
                corr_matrix_pd,
                mask=mask | corr_matrix_pd.isna(),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
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
            ax.set_title(
                "Spearman Rank Correlation Between Evaluation Metrics"
                if corr_method == "spearman"
                else "Pearson Correlation Between Evaluation Metrics"
            )
            fig.tight_layout()

        formatter = OUTPUT_FORMATTERS[self.output_format]
        return cast(
            T,
            CorrelationResults(
                correlation_matrix=formatter(corr_matrix),
                figure=fig,
                paired_counts=formatter(count_matrix),
                excluded_counts=formatter(excluded_matrix),
            ),
        )
