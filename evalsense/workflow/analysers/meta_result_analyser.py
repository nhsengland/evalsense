from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Literal, cast, override
import warnings

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import binomtest, bootstrap, spearmanr

from evalsense.evaluation import MetaTierGroupedRecord
from evalsense.workflow import Project, ResultAnalyser

OUTPUT_FORMATTERS = {
    "polars": lambda df: df,
    "pandas": lambda df: df.to_pandas(),
}

type SampleKey = tuple[str, str | int]
type MetricKey = tuple[str, str | None]


@dataclass
class _Observation:
    scores: dict[MetricKey, Any]
    usable: bool
    validated: bool


@dataclass
class _Cohort:
    samples: dict[SampleKey, dict[int, _Observation]] = field(default_factory=dict)
    metrics: set[MetricKey] = field(default_factory=set)
    epoch: int | None = None
    failed_logs: int = 0
    unavailable_logs: int = 0


_SCHEMA = {
    "group_id": pl.String,
    "dataset": pl.String,
    "model": pl.String,
    "task": pl.String,
    "evaluator": pl.String,
    "scorer": pl.String,
    "metric_name": pl.String,
    "analysis_kind": pl.String,
    "pairwise_accuracy": pl.Float64,
    "n_concordant": pl.Int64,
    "n_discordant": pl.Int64,
    "avg_correlation": pl.Float64,
    "ci_low": pl.Float64,
    "ci_high": pl.Float64,
    "ci_status": pl.String,
    "n_samples": pl.Int64,
    "n_included": pl.Int64,
    "n_excluded": pl.Int64,
    "n_incomplete_tiers": pl.Int64,
    "n_missing_scores": pl.Int64,
    "n_invalid_scores": pl.Int64,
    "n_unvalidated": pl.Int64,
    "n_constant": pl.Int64,
    "n_undefined": pl.Int64,
    "failed_logs": pl.Int64,
    "unavailable_logs": pl.Int64,
    "sample_population": pl.String,
    "expected_tiers": pl.String,
    "lower_tier_is_better": pl.Boolean,
    "validation_policy": pl.String,
    "tier_validation_field": pl.String,
    "confidence_level": pl.Float64,
    "n_resamples": pl.Int64,
    "seed": pl.Int64,
}


def _sample_key(value: str | int) -> SampleKey:
    if type(value) not in (str, int):
        raise ValueError("Sample IDs must be strings or integers, not booleans.")
    return (type(value).__name__, value)


def _tier_set(expected_tiers: Sequence[int]) -> tuple[int, ...]:
    if isinstance(expected_tiers, (str, bytes)):
        raise ValueError("expected_tiers must contain at least two distinct integers.")
    tiers = tuple(expected_tiers)
    if any(type(tier) is not int for tier in tiers) or len(set(tiers)) != len(tiers):
        raise ValueError("expected_tiers must contain distinct integer tier IDs.")
    if len(tiers) < 2:
        raise ValueError("Meta-evaluation requires at least two expected tiers.")
    return tuple(sorted(tiers))


def _interval(
    correlations: list[float], confidence_level: float, n_resamples: int, seed: int
) -> tuple[float | None, float | None, str]:
    if len(correlations) < 2:
        return None, None, "insufficient_samples"
    if len(set(correlations)) == 1:
        # A zero-width empirical interval would conceal the lack of variation.
        return None, None, "degenerate"
    result = bootstrap(
        (np.asarray(correlations, dtype=float),),
        np.mean,
        vectorized=True,
        method="percentile",
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        batch=128,
        rng=np.random.default_rng(seed),
    )
    low, high = map(float, result.confidence_interval)
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        return None, None, "degenerate"
    return low, high, "percentile_bootstrap"


def _pairwise_interval(
    concordant: int, total: int, confidence_level: float
) -> tuple[float | None, float | None, str]:
    """Wilson limits for non-tied ordering accuracy, mapped to the mean-rho scale."""
    if total == 0:
        return None, None, "insufficient_samples"
    interval = binomtest(concordant, total).proportion_ci(
        confidence_level=confidence_level, method="wilson"
    )
    # With two distinct scores rho is +1 or -1, so its mean is 2 * accuracy - 1.
    # Wilson limits remain informative when every observed pair has the same sign.
    return 2 * float(interval.low) - 1, 2 * float(interval.high) - 1, "wilson_pairwise"


class MetaResultAnalyser[T: pl.DataFrame | pd.DataFrame](ResultAnalyser[T]):
    """Describe within-sample rank agreement across a declared complete tier set.

    For three or more planned tiers, each sample contributes one Spearman
    correlation. A deliberately two-tier design instead reports pairwise ordering:
    ``pairwise_accuracy`` is concordant / (concordant + discordant), and
    ``avg_correlation`` is 2 * pairwise_accuracy - 1. ``analysis_kind`` identifies
    the design. Equal-score pairs are undefined correlations, excluded and counted
    in ``n_constant``; pairwise accuracy is conditional on non-tied, usable pairs.
    A missing tier never changes the declared design or selects pairwise analysis.

    The reported mean gives included samples equal weight. Results are kept
    separate by dataset, task, generating model, evaluator configuration and scorer;
    only the tier generator is grouped out. ``group_id`` is a stable digest of
    that grouped record; full configurations remain in the project. Aliases
    change presentation only.

    Counts cover the union of sample IDs in available evaluation logs, including
    failed logs, or an explicit ``expected_sample_ids`` manifest. Samples absent
    from every log cannot be counted without that manifest. Missing logs are
    reported separately. Scores from unsuccessful records/logs are not used.
    Exclusion reasons are mutually exclusive, checked in this order: incomplete
    tiers, missing/failed scores, invalid/non-finite values, missing validation,
    constant scores, undefined correlation. No score is imputed or set to zero.

    For three or more tiers, confidence limits use a percentile bootstrap of
    included sample-level correlations, never of individual tiers. Those intervals
    are unavailable for fewer than two included samples or degenerate distributions.
    Two-tier designs use Wilson binomial limits for the ordering accuracy, mapped
    to the mean-correlation scale as 2 * limit - 1. They are defined even for one
    non-tied pair or unanimous pairs; no usable pairs gives no estimate or interval.
    ``ci_low`` and ``ci_high`` are always on the ``avg_correlation`` scale.

    Both interval methods assume independent sample IDs and are conditional on
    retained samples and the tier design. They do not correct selection bias,
    establish clinical validity, or quantify uncertainty in generated severity.
    Small samples and few tiers provide limited evidence; a perfect mean alone
    is not evidence of a reliable evaluator.

    Generated tier IDs specify intended severity, not independently verified
    severity. By default results carry ``validation_policy='not_checked'`` and a
    warning. ``tier_validation_field`` requires caller-supplied True flags on all
    tiers of a sample, after independent review of the actual generated outputs.
    These flags are attestations, not validation performed by this analyser.

    Duplicate sample/tier observations and multiple epochs within a cohort raise
    ValueError. Repeated measurements need a separate, explicit aggregation policy.
    """

    def __init__(
        self,
        name: str = "MetaResultAnalyser",
        output_format: Literal["polars", "pandas"] = "polars",
    ):
        """Initialise the analyser.

        Args:
            name (str): Name of the analyser.
            output_format (Literal["polars", "pandas"]): Result table format.

        Raises:
            ValueError: If the output format is unsupported.
        """
        super().__init__(name=name)
        if output_format not in OUTPUT_FORMATTERS:
            raise ValueError(
                f"Invalid output format: {output_format}. Use polars or pandas."
            )
        self.output_format = output_format

    @override
    def __call__(
        self,
        project: Project,
        meta_tier_field: str = "perturbation_type_tier",
        lower_tier_is_better: bool = False,
        metric_labels: dict[str, str] | None = None,
        *,
        expected_tiers: Sequence[int],
        expected_sample_ids: Sequence[str | int] | None = None,
        tier_validation_field: str | None = None,
        confidence_level: float = 0.95,
        n_resamples: int = 2000,
        seed: int = 0,
        **kwargs: dict,
    ) -> T:
        """Summarise complete, finite tier comparisons with explicit evidence counts.

        Args:
            project (Project): Project containing the evaluation logs.
            meta_tier_field (str): Metadata field containing integer tier IDs.
            lower_tier_is_better (bool): True when lower tiers mean better outputs.
                All analysed metrics must be oriented so higher scores are better.
            metric_labels (dict[str, str] | None): Display aliases for original
                scalar scorer or inner metric names. Does not combine identities.
            expected_tiers (Sequence[int]): Complete planned tier set, with at least
                two distinct integers. Two tiers select pairwise-ordering analysis;
                three or more select mean Spearman correlation. Never inferred
                from surviving results.
            expected_sample_ids (Sequence[str | int] | None): Optional full sample
                manifest, applied to each cohort. IDs must be unique and cover all
                observations. Without it, counts describe observed IDs only.
            tier_validation_field (str | None): Optional metadata field containing
                the boolean True after external review of each generated output's
                assigned tier. Missing/false flags exclude the whole sample.
            confidence_level (float): Two-sided confidence level in (0, 1).
            n_resamples (int): Bootstrap resamples for three or more tiers, at least
                1000. Defaults to 2000. Unused for two-tier designs.
            seed (int): Non-negative bootstrap seed. Defaults to 0. Unused for
                two-tier designs.
            **kwargs (dict): Unused arguments accepted for ResultAnalyser compatibility.

        Returns:
            T: One row per cohort/scorer/metric, with the mean correlation, interval,
                interval status, included/excluded counts, exclusion reasons and
                analysis settings. Two-tier rows include concordant/discordant
                counts and accuracy; bootstrap-only settings are null. Unestimable
                values are null. Cohorts without any recorded metrics have a null
                metric row. Empty projects return
                a typed empty table. Neither logs nor project records are modified.

        Raises:
            ValueError: For invalid settings, invalid/unexpected tiers, IDs outside
                the manifest, duplicate sample/tier observations or mixed epochs.
        """
        tiers = _tier_set(expected_tiers)
        pairwise = len(tiers) == 2
        if not meta_tier_field or not isinstance(meta_tier_field, str):
            raise ValueError("meta_tier_field must be a non-empty string.")
        if type(lower_tier_is_better) is not bool:
            raise ValueError("lower_tier_is_better must be a boolean.")
        if type(confidence_level) not in (int, float) or not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one.")
        if type(n_resamples) is not int or n_resamples < 1000:
            raise ValueError("n_resamples must be an integer of at least 1000.")
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be a non-negative 64-bit integer.")
        if tier_validation_field is not None and (
            not isinstance(tier_validation_field, str)
            or not tier_validation_field.strip()
        ):
            raise ValueError("tier_validation_field must be a non-empty string.")
        population = None
        if expected_sample_ids is not None:
            if isinstance(expected_sample_ids, (str, bytes)):
                raise ValueError(
                    "expected_sample_ids must be a sequence of sample IDs."
                )
            population = {_sample_key(value) for value in expected_sample_ids}
            if len(population) != len(expected_sample_ids):
                raise ValueError("expected_sample_ids must not contain duplicates.")
        if tier_validation_field is None:
            warnings.warn(
                "Meta-evaluation tier severity has not been independently checked. "
                "Interpret these correlations as agreement with intended tiers, "
                "not validated evaluator accuracy.",
                UserWarning,
                stacklevel=2,
            )

        cohorts: dict[MetaTierGroupedRecord, _Cohort] = {}
        logs = project.get_logs(type="evaluation")
        for record, result in project.records.evaluation.items():
            group = record.get_meta_grouped_record("")
            cohort = cohorts.setdefault(group, _Cohort())
            log = logs.get(record)
            if result.status != "success" or (
                log is not None and log.status != "success"
            ):
                cohort.failed_logs += 1
            if log is None or not log.samples:
                cohort.unavailable_logs += 1
                continue
            for sample in log.samples:
                tier = sample.metadata.get(meta_tier_field)
                if type(tier) is not int or tier not in tiers:
                    raise ValueError(
                        f"Metadata field {meta_tier_field!r} must contain an expected "
                        "integer tier ID; check the experiment design and log metadata."
                    )
                if cohort.epoch is not None and cohort.epoch != sample.epoch:
                    raise ValueError(
                        "Multiple epochs require an explicit aggregation policy. "
                        "Analyse one epoch at a time."
                    )
                cohort.epoch = sample.epoch
                key = _sample_key(sample.id)
                if population is not None and key not in population:
                    raise ValueError(
                        "Observed sample ID is absent from expected_sample_ids."
                    )
                observations = cohort.samples.setdefault(key, {})
                if tier in observations:
                    raise ValueError(
                        "Duplicate sample/tier observation; results cannot be overwritten."
                    )
                values: dict[MetricKey, Any] = {}
                for scorer_name, score in (sample.scores or {}).items():
                    if isinstance(score.value, dict):
                        values.update(
                            {
                                (scorer_name, inner): value
                                for inner, value in score.value.items()
                            }
                        )
                    else:
                        values[(scorer_name, None)] = score.value
                cohort.metrics.update(values)
                observations[tier] = _Observation(
                    values,
                    result.status == log.status == "success" and sample.error is None,
                    tier_validation_field is not None
                    and sample.metadata.get(tier_validation_field) is True,
                )

        rows = []
        for group, cohort in sorted(
            cohorts.items(), key=lambda item: item[0].model_dump_json()
        ):
            group_json = json.dumps(group.model_dump(mode="json"), sort_keys=True)
            sample_ids = sorted(
                population if population is not None else cohort.samples
            )
            metrics = sorted(cohort.metrics, key=lambda key: (key[0], key[1] or ""))
            for metric in metrics or [None]:
                counts: Counter[str] = Counter()
                correlations = []
                for sample_id in sample_ids:
                    observed = cohort.samples.get(sample_id, {})
                    if set(observed) != set(tiers):
                        counts["incomplete_tiers"] += 1
                        continue
                    if metric is None or any(
                        not observed[t].usable or metric not in observed[t].scores
                        for t in tiers
                    ):
                        counts["missing_scores"] += 1
                        continue
                    tier_values = [observed[t].scores[metric] for t in tiers]
                    if any(
                        type(v) not in (int, float) or not math.isfinite(v)
                        for v in tier_values
                    ):
                        counts["invalid_scores"] += 1
                        continue
                    if tier_validation_field is not None and not all(
                        observed[t].validated for t in tiers
                    ):
                        counts["unvalidated"] += 1
                        continue
                    if len(set(tier_values)) == 1:
                        counts["constant"] += 1
                        continue
                    multiplier = -1 if lower_tier_is_better else 1
                    if pairwise:
                        # Compare directly to avoid overflow when subtracting scores.
                        correlation = (
                            float(multiplier)
                            if tier_values[1] > tier_values[0]
                            else -float(multiplier)
                        )
                    else:
                        correlation = float(
                            cast(
                                float,
                                spearmanr([multiplier * t for t in tiers], tier_values)[
                                    0
                                ],
                            )
                        )
                    if not math.isfinite(correlation):
                        counts["undefined"] += 1
                        continue
                    correlations.append(correlation)
                concordant = (
                    sum(value > 0 for value in correlations) if pairwise else None
                )
                if pairwise:
                    assert concordant is not None
                    low, high, interval_status = _pairwise_interval(
                        concordant, len(correlations), confidence_level
                    )
                else:
                    low, high, interval_status = _interval(
                        correlations, confidence_level, n_resamples, seed
                    )
                raw_name = (
                    (metric[1] if metric[1] is not None else metric[0])
                    if metric
                    else None
                )
                rows.append(
                    {
                        "group_id": hashlib.sha256(group_json.encode()).hexdigest(),
                        "dataset": group.dataset_record.name,
                        "model": group.model_record.name,
                        "task": group.task_name,
                        "evaluator": group.evaluator_name,
                        "scorer": metric[0] if metric else None,
                        "metric_name": (metric_labels or {}).get(raw_name, raw_name)
                        if raw_name is not None
                        else None,
                        "analysis_kind": "pairwise_ordering"
                        if pairwise
                        else "mean_spearman",
                        "pairwise_accuracy": concordant / len(correlations)
                        if concordant is not None and correlations
                        else None,
                        "n_concordant": concordant,
                        "n_discordant": len(correlations) - concordant
                        if concordant is not None
                        else None,
                        "avg_correlation": math.fsum(correlations) / len(correlations)
                        if correlations
                        else None,
                        "ci_low": low,
                        "ci_high": high,
                        "ci_status": interval_status,
                        "n_samples": len(sample_ids),
                        "n_included": len(correlations),
                        "n_excluded": len(sample_ids) - len(correlations),
                        **{
                            f"n_{reason}": counts[reason]
                            for reason in (
                                "incomplete_tiers",
                                "missing_scores",
                                "invalid_scores",
                                "unvalidated",
                                "constant",
                                "undefined",
                            )
                        },
                        "failed_logs": cohort.failed_logs,
                        "unavailable_logs": cohort.unavailable_logs,
                        "sample_population": "declared"
                        if population is not None
                        else "observed",
                        "expected_tiers": json.dumps(tiers),
                        "lower_tier_is_better": lower_tier_is_better,
                        "validation_policy": "require_caller_attestation"
                        if tier_validation_field
                        else "not_checked",
                        "tier_validation_field": tier_validation_field,
                        "confidence_level": float(confidence_level),
                        "n_resamples": None if pairwise else n_resamples,
                        "seed": None if pairwise else seed,
                    }
                )
        return cast(
            T, OUTPUT_FORMATTERS[self.output_format](pl.DataFrame(rows, schema=_SCHEMA))
        )
