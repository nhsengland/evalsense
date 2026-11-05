"""Deterministic tier, denominator and uncertainty regressions without model calls."""

import json
import math
from statistics import NormalDist
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import polars as pl
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalSample,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.scorer import Score

from evalsense.datasets import DatasetRecord
from evalsense.evaluation import EvaluationRecord, ResultRecord
from evalsense.generation import ModelRecord
from evalsense.workflow import Project
from evalsense.workflow.analysers import MetaResultAnalyser
from evalsense.workflow.analysers.meta_result_analyser import _interval


TIERS = (0, 1, 2)


def record(tier: int, **kwargs: Any) -> EvaluationRecord:
    values: dict[str, Any] = dict(
        dataset_record=DatasetRecord(name="data", version="v1", splits=("test",)),
        model_record=ModelRecord(name="model"),
        task_name="task",
        generator_name=f"tier {tier}",
        evaluator_name="judge",
    )
    values.update(kwargs)
    return EvaluationRecord(**values)


def sample(id: str | int, tier: Any, value: Any, **kwargs: Any) -> EvalSample:
    return EvalSample(
        id=id,
        epoch=kwargs.pop("epoch", 1),
        input="input",
        target="target",
        metadata=kwargs.pop("metadata", {"perturbation_type_tier": tier}),
        scores=kwargs.pop("scores", {"quality": Score(value=value)}),
        **kwargs,
    )


def log(samples: list[EvalSample], **kwargs: Any) -> EvalLog:
    return EvalLog(
        status=kwargs.pop("status", "success"),
        eval=EvalSpec(
            created="2026-01-01T00:00:00Z",
            task="task",
            model="model",
            dataset=EvalDataset(),
            config=EvalConfig(),
        ),
        samples=samples,
        **kwargs,
    )


def project(logs: dict[EvaluationRecord, EvalLog], states=None) -> Project:
    result = Mock(spec=Project)
    result.records = SimpleNamespace(
        evaluation={
            key: ResultRecord(status=(states or {}).get(key, "success")) for key in logs
        }
    )
    result.get_logs.return_value = logs
    return cast(Project, result)


def complete(values: dict[str | int, list[Any]], **record_options: Any) -> Project:
    return project(
        {
            record(tier, **record_options): log(
                [sample(id, tier, scores[tier]) for id, scores in values.items()]
            )
            for tier in TIERS
        }
    )


def analyse(data: Project, **kwargs: Any) -> pl.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        options: dict[str, Any] = {"expected_tiers": TIERS} | kwargs
        return MetaResultAnalyser[pl.DataFrame]()(data, **options)


class MetaEvidenceTests(unittest.TestCase):
    def test_two_observed_tiers_cannot_report_perfect_agreement(self):
        data = project({record(t): log([sample("one", t, t)]) for t in (0, 1)})
        row = analyse(data).row(0, named=True)
        self.assertIsNone(row["avg_correlation"])
        self.assertEqual(row["n_incomplete_tiers"], 1)
        self.assertEqual(row["n_included"], 0)
        self.assertEqual(row["n_excluded"], 1)
        self.assertEqual(row["ci_status"], "insufficient_samples")

    def test_expected_tiers_are_required_and_not_inferred(self):
        with self.assertRaises(TypeError):
            MetaResultAnalyser()(complete({1: [0, 1, 2]}))  # type: ignore[call-arg]
        for tiers in [
            [],
            [0],
            [0, 0],
            [0, True],
            [0, 1.0],
            [0, 1, 1],
            [0, 1, True],
            [0, 1, 2.0],
            "012",
        ]:
            with self.subTest(tiers=tiers), self.assertRaises(ValueError):
                analyse(complete({1: [0, 1, 2]}), expected_tiers=tiers)

    def test_declared_but_entirely_absent_tier_excludes_all_samples(self):
        row = analyse(
            complete({1: [0, 1, 2], 2: [0, 2, 1]}), expected_tiers=[0, 1, 2, 3]
        ).row(0, named=True)
        self.assertEqual(row["n_incomplete_tiers"], 2)
        self.assertIsNone(row["avg_correlation"])

    def test_counts_reconcile_without_hiding_constant_or_missing_scores(self):
        logs = {}
        for tier in TIERS:
            samples = [
                sample("good", tier, tier),
                sample("constant", tier, 1),
                sample("nan", tier, math.nan if tier == 2 else tier),
                sample(
                    "missing_score",
                    tier,
                    tier,
                    scores={} if tier == 2 else {"quality": Score(value=tier)},
                ),
            ]
            if tier != 2:
                samples.append(sample("incomplete", tier, tier))
            logs[record(tier)] = log(samples)
        row = analyse(project(logs)).row(0, named=True)
        self.assertEqual(row["n_samples"], 5)
        self.assertEqual(row["n_included"], 1)
        self.assertEqual(row["n_excluded"], 4)
        self.assertEqual(row["avg_correlation"], 1)
        for reason in (
            "incomplete_tiers",
            "missing_scores",
            "invalid_scores",
            "constant",
        ):
            self.assertEqual(row[f"n_{reason}"], 1)
        self.assertEqual(
            sum(
                row[f"n_{r}"]
                for r in (
                    "incomplete_tiers",
                    "missing_scores",
                    "invalid_scores",
                    "constant",
                    "undefined",
                    "unvalidated",
                )
            ),
            row["n_excluded"],
        )
        self.assertIsNone(row["ci_low"])

    def test_constant_cohort_is_returned_as_unestimable(self):
        row = analyse(complete({1: [1, 1, 1]})).row(0, named=True)
        self.assertIsNone(row["avg_correlation"])
        self.assertEqual(row["n_constant"], 1)
        self.assertEqual(row["n_samples"], 1)

    def test_non_finite_and_nonnumeric_values_are_explicit_exclusions(self):
        for value in (math.nan, math.inf, -math.inf, "C", True, [1, 2]):
            with self.subTest(value=value):
                row = analyse(complete({1: [0, 1, value]})).row(0, named=True)
                self.assertEqual(row["n_invalid_scores"], 1)
                self.assertIsNone(row["avg_correlation"])

    def test_tier_direction_is_explicit_and_ties_are_ranked(self):
        data = complete({1: [3, 2, 1]})
        self.assertEqual(analyse(data)["avg_correlation"].item(), -1)
        self.assertEqual(
            analyse(data, lower_tier_is_better=True)["avg_correlation"].item(), 1
        )
        result = analyse(complete({1: [1, 1, 2]}))["avg_correlation"].item()
        self.assertAlmostEqual(result, math.sqrt(3) / 2)

    def test_nonconsecutive_tiers_are_ranked_numerically(self):
        tiers = [-5, 10, 100]
        logs = {record(t): log([sample(1, t, i)]) for i, t in enumerate(tiers)}
        self.assertEqual(
            analyse(project(logs), expected_tiers=tiers[::-1])[
                "avg_correlation"
            ].item(),
            1,
        )

    def test_bad_metadata_is_rejected_instead_of_coerced(self):
        for tier in (None, "1", 1.2, True, 4):
            with (
                self.subTest(tier=tier),
                self.assertRaisesRegex(ValueError, "integer tier"),
            ):
                analyse(project({record(0): log([sample(1, tier, 1)])}))

    def test_perfect_sample_correlations_do_not_get_zero_width_intervals(self):
        for count in (1, 2, 20):
            row = analyse(complete({i: [0, 1, 2] for i in range(count)})).row(
                0, named=True
            )
            self.assertEqual(row["avg_correlation"], 1)
            self.assertIsNone(row["ci_low"])
            self.assertIsNone(row["ci_high"])
            self.assertEqual(
                row["ci_status"], "insufficient_samples" if count == 1 else "degenerate"
            )

    def test_bootstrap_resamples_samples_not_tiers(self):
        row = analyse(complete({1: [0, 1, 2], 2: [2, 1, 0]})).row(0, named=True)
        self.assertEqual(row["avg_correlation"], 0)
        self.assertEqual(row["n_included"], 2)
        self.assertEqual((row["ci_low"], row["ci_high"]), (-1, 1))
        self.assertEqual(row["ci_status"], "percentile_bootstrap")

    def test_bootstrap_matches_independent_reference_resampling(self):
        values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        seed, count = 42, 2000
        indices = np.random.default_rng(seed).integers(
            len(values), size=(count, len(values))
        )
        distribution = np.asarray(values)[indices].mean(axis=1)
        expected = np.quantile(distribution, [0.025, 0.975])
        actual = _interval(values, 0.95, count, seed)
        assert actual[0] is not None and actual[1] is not None
        np.testing.assert_allclose([actual[0], actual[1]], expected, rtol=0, atol=1e-15)
        self.assertEqual(actual[2], "percentile_bootstrap")

    def test_bootstrap_is_seeded_and_does_not_change_global_rng(self):
        data = complete({1: [0, 1, 2], 2: [2, 0, 1], 3: [2, 1, 0]})
        state = cast(tuple[Any, ...], np.random.get_state(legacy=True))
        first, second = analyse(data, seed=19), analyse(data, seed=19)
        self.assertTrue(first.equals(second))
        after = cast(tuple[Any, ...], np.random.get_state(legacy=True))
        np.testing.assert_equal(after[1], state[1])
        self.assertEqual(after[2:], state[2:])

    def test_log_sample_and_tier_order_do_not_change_result(self):
        data = complete({3: [0, 1, 2], 1: [2, 0, 1], 2: [2, 1, 0]})
        first = analyse(data)
        logs = {
            key: value.model_copy(
                update={"samples": list(reversed(value.samples or []))}
            )
            for key, value in reversed(list(data.get_logs(type="evaluation").items()))
        }
        self.assertTrue(first.equals(analyse(project(logs), expected_tiers=[2, 0, 1])))

    def test_validation_is_not_assumed_from_generated_labels(self):
        data = complete({1: [0, 1, 2]})
        with self.assertWarnsRegex(UserWarning, "not been independently checked"):
            table = MetaResultAnalyser[pl.DataFrame]()(data, expected_tiers=TIERS)
        self.assertEqual(table["validation_policy"].item(), "not_checked")
        self.assertIsNone(table["tier_validation_field"].item())

    def test_external_validation_requires_true_for_every_tier(self):
        logs: dict[EvaluationRecord, EvalLog] = {}
        for flag in (False, "true", 1, None):
            logs = {
                record(t): log(
                    [
                        sample(
                            1,
                            t,
                            t,
                            metadata={
                                "perturbation_type_tier": t,
                                "reviewed": flag if t == 2 else True,
                            },
                        )
                    ]
                )
                for t in TIERS
            }
            with self.subTest(flag=flag):
                row = analyse(project(logs), tier_validation_field="reviewed").row(
                    0, named=True
                )
                self.assertEqual(row["n_unvalidated"], 1)
                self.assertIsNone(row["avg_correlation"])
        for value in logs.values():
            assert value.samples is not None
            value.samples[0].metadata["reviewed"] = True
        row = analyse(project(logs), tier_validation_field="reviewed").row(
            0, named=True
        )
        self.assertEqual(row["n_included"], 1)
        self.assertEqual(row["validation_policy"], "require_caller_attestation")

    def test_independent_annotation_field_can_supply_tiers(self):
        logs = {
            record(t): log(
                [sample(1, t, t, metadata={"adjudicated_tier": t, "reviewed": True})]
            )
            for t in TIERS
        }
        with warnings.catch_warnings(record=True) as caught:
            table = MetaResultAnalyser[pl.DataFrame]()(
                project(logs),
                expected_tiers=TIERS,
                meta_tier_field="adjudicated_tier",
                tier_validation_field="reviewed",
            )
        self.assertEqual(len(caught), 0)
        self.assertEqual(table["avg_correlation"].item(), 1)

    def test_sample_manifest_counts_entirely_absent_samples(self):
        row = analyse(complete({1: [0, 1, 2]}), expected_sample_ids=[1, 2, "1"]).row(
            0, named=True
        )
        self.assertEqual(row["n_samples"], 3)
        self.assertEqual(row["n_incomplete_tiers"], 2)
        self.assertEqual(row["sample_population"], "declared")
        with self.assertRaisesRegex(ValueError, "absent"):
            analyse(complete({1: [0, 1, 2]}), expected_sample_ids=[2])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            analyse(complete({1: [0, 1, 2]}), expected_sample_ids=[1, 1])

    def test_string_and_integer_sample_ids_are_distinct(self):
        row = analyse(complete({1: [0, 1, 2], "1": [2, 1, 0]})).row(0, named=True)
        self.assertEqual(row["n_included"], 2)
        self.assertEqual(row["avg_correlation"], 0)

    def test_duplicates_are_rejected_even_when_scores_equal(self):
        logs = complete({1: [0, 1, 2]}).get_logs(type="evaluation")
        samples = logs[record(0)].samples
        assert samples is not None
        samples.append(sample(1, 0, 0))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyse(project(logs))

    def test_repeated_or_mixed_epochs_cannot_overwrite_results(self):
        for reverse in (False, True):
            observations = [sample(1, 0, 0), sample(1, 0, 1, epoch=2)]
            if reverse:
                observations.reverse()
            with (
                self.subTest(reverse=reverse),
                self.assertRaisesRegex(ValueError, "Multiple epochs"),
            ):
                analyse(project({record(0): log(observations)}))
        logs = complete({1: [0, 1, 2]}).get_logs(type="evaluation")
        samples = logs[record(2)].samples
        assert samples is not None
        samples[0].epoch = 2
        with self.assertRaisesRegex(ValueError, "Multiple epochs"):
            analyse(project(logs))

    def test_failed_logs_supply_population_but_not_scores(self):
        logs = complete({1: [0, 1, 2]}).get_logs(type="evaluation")
        data = project(logs, states={record(2): "error"})
        row = analyse(data).row(0, named=True)
        self.assertEqual(row["failed_logs"], 1)
        self.assertEqual(row["n_missing_scores"], 1)
        self.assertIsNone(row["avg_correlation"])
        logs[record(1)].status = "cancelled"
        row = analyse(data).row(0, named=True)
        self.assertEqual(row["failed_logs"], 2)

    def test_missing_log_is_reported_without_assuming_a_smaller_design(self):
        logs = complete({1: [0, 1, 2]}).get_logs(type="evaluation")
        data = project(logs)
        del logs[record(2)]
        row = analyse(data).row(0, named=True)
        self.assertEqual(row["unavailable_logs"], 1)
        self.assertEqual(row["n_incomplete_tiers"], 1)
        self.assertEqual(row["n_included"], 0)

    def test_no_recorded_scores_or_no_available_logs_still_produce_diagnostics(self):
        logs = {record(t): log([sample(1, t, 1, scores={})]) for t in TIERS}
        row = analyse(project(logs)).row(0, named=True)
        self.assertIsNone(row["metric_name"])
        self.assertEqual(row["n_missing_scores"], 1)
        data = project(logs)
        logs.clear()
        row = analyse(data, expected_sample_ids=[1, 2]).row(0, named=True)
        self.assertEqual(row["unavailable_logs"], 3)
        self.assertEqual(row["n_incomplete_tiers"], 2)
        self.assertIsNone(row["metric_name"])

    def test_distinct_experiment_configurations_are_not_pooled(self):
        differences = [
            dict(model_record=ModelRecord(name="other")),
            dict(task_name="other"),
            dict(
                dataset_record=DatasetRecord(
                    name="data", version="v2", splits=("test",)
                )
            ),
            dict(evaluator_name="other"),
            dict(experiment_name="other"),
        ]
        for change in differences:
            with self.subTest(change=change):
                a = complete({1: [0, 1, 2]}).get_logs(type="evaluation")
                b = complete({1: [2, 1, 0]}, **change).get_logs(type="evaluation")
                rows = analyse(project(a | b))
                self.assertEqual(rows.height, 2)
                self.assertEqual(sorted(rows["avg_correlation"].to_list()), [-1, 1])
                self.assertEqual(rows["group_id"].n_unique(), 2)

    def test_scorers_inner_metrics_and_display_aliases_do_not_merge(self):
        logs = {
            record(t): log(
                [
                    sample(
                        1,
                        t,
                        t,
                        scores={
                            "left": Score(value={"quality": t}),
                            "right": Score(value={"quality": 2 - t}),
                        },
                    )
                ]
            )
            for t in TIERS
        }
        rows = analyse(project(logs), metric_labels={"quality": "same"})
        self.assertEqual(rows.height, 2)
        self.assertEqual(rows["metric_name"].to_list(), ["same", "same"])
        self.assertEqual(sorted(rows["avg_correlation"].to_list()), [-1, 1])
        self.assertEqual(rows["scorer"].n_unique(), 2)

    def test_empty_project_has_stable_schema_in_both_formats(self):
        for output in ("polars", "pandas"):
            with self.subTest(output=output), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rows = MetaResultAnalyser(output_format=cast(Any, output))(
                    project({}), expected_tiers=TIERS
                )
                self.assertEqual(len(rows), 0)
                self.assertIn("avg_correlation", rows.columns)
                self.assertIn("n_samples", rows.columns)
                self.assertIn("ci_status", rows.columns)

    def test_invalid_settings_fail_before_reading_logs(self):
        settings = [{"confidence_level": x} for x in (0, 1, math.nan, math.inf, True)]
        settings += [{"n_resamples": x} for x in (0, 10, True, 1000.0)]
        settings += [{"seed": x} for x in (-1, True, 1.2)]
        settings += [
            {"tier_validation_field": ""},
            {"meta_tier_field": ""},
            {"lower_tier_is_better": "yes"},
            {"expected_sample_ids": "abc"},
        ]
        for options in settings:
            data = project({})
            with self.subTest(options=options), self.assertRaises(ValueError):
                analyse(data, **options)
            cast(Any, data.get_logs).assert_not_called()

    def test_project_log_roundtrip_and_read_only_analysis(self):
        for suffix in ("json", "eval"):
            with (
                self.subTest(suffix=suffix),
                tempfile.TemporaryDirectory() as tmp,
                patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp)),
            ):
                stored = Project("meta")
                stored.evaluation_log_path.mkdir(parents=True)
                for t in TIERS:
                    path = stored.evaluation_log_path / f"2026_task_{t}.{suffix}"
                    write_eval_log(
                        log([sample(1, t, t), sample(2, t, 2 - t)]), location=str(path)
                    )
                    stored.update_record(
                        record(t),
                        ResultRecord(status="success", log_location=str(path)),
                    )
                before = {
                    str(p): p.read_bytes() for p in Path(tmp).rglob("*") if p.is_file()
                }
                reopened = Project("meta")
                rows = analyse(reopened)
                after = {
                    str(p): p.read_bytes() for p in Path(tmp).rglob("*") if p.is_file()
                }
                self.assertEqual(before, after)
                self.assertEqual(rows["n_included"].item(), 2)
                self.assertEqual(rows["avg_correlation"].item(), 0)
                self.assertEqual(json.loads(rows["expected_tiers"].item()), [0, 1, 2])


def two_tier(values: dict[str | int, list[Any]], tiers=(0, 1)) -> Project:
    return project(
        {
            record(tier): log(
                [sample(id, tier, scores[index]) for id, scores in values.items()]
            )
            for index, tier in enumerate(tiers)
        }
    )


class TwoTierMetaTests(unittest.TestCase):
    def test_planned_two_tiers_measure_ordering_across_samples(self):
        data = two_tier({i: [0, 1] if i < 80 else [1, 0] for i in range(100)})
        row = analyse(data, expected_tiers=(0, 1)).row(0, named=True)
        self.assertEqual(row["analysis_kind"], "pairwise_ordering")
        self.assertEqual(row["n_included"], 100)
        self.assertEqual(row["n_concordant"], 80)
        self.assertEqual(row["n_discordant"], 20)
        self.assertEqual(row["pairwise_accuracy"], 0.8)
        self.assertAlmostEqual(row["avg_correlation"], 0.6)
        self.assertEqual(row["ci_status"], "wilson_pairwise")
        self.assertIsNone(row["n_resamples"])
        self.assertIsNone(row["seed"])

    def test_wilson_limits_match_independent_formula_on_mean_scale(self):
        for successes, total, confidence in [
            (80, 100, 0.95),
            (0, 12, 0.9),
            (12, 12, 0.99),
            (1, 1, 0.95),
        ]:
            with self.subTest(successes=successes, total=total, confidence=confidence):
                data = two_tier(
                    {i: [0, 1] if i < successes else [1, 0] for i in range(total)}
                )
                row = analyse(
                    data, expected_tiers=(0, 1), confidence_level=confidence
                ).row(0, named=True)
                z = NormalDist().inv_cdf((1 + confidence) / 2)
                p = successes / total
                denominator = 1 + z * z / total
                centre = (p + z * z / (2 * total)) / denominator
                radius = (
                    z
                    * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
                    / denominator
                )
                self.assertAlmostEqual(
                    row["ci_low"], 2 * (centre - radius) - 1, places=12
                )
                self.assertAlmostEqual(
                    row["ci_high"], 2 * (centre + radius) - 1, places=12
                )
                self.assertLess(row["ci_low"], row["ci_high"])

    def test_direction_and_nonconsecutive_tiers_are_explicit(self):
        data = two_tier({1: [3, 1], 2: [4, 2], 3: [1, 2]}, tiers=(-5, 10))
        ascending = analyse(data, expected_tiers=(10, -5)).row(0, named=True)
        descending = analyse(
            data, expected_tiers=(-5, 10), lower_tier_is_better=True
        ).row(0, named=True)
        self.assertEqual(ascending["n_concordant"], 1)
        self.assertEqual(descending["n_concordant"], 2)
        self.assertAlmostEqual(
            ascending["avg_correlation"], -descending["avg_correlation"]
        )
        self.assertAlmostEqual(ascending["ci_low"], -descending["ci_high"])

    def test_ties_are_counted_and_not_used_as_zero_correlations(self):
        data = two_tier({"yes": [0, 1], "no": [1, 0], "tie": [2, 2]})
        row = analyse(data, expected_tiers=(0, 1)).row(0, named=True)
        self.assertEqual(row["n_samples"], 3)
        self.assertEqual(row["n_included"], 2)
        self.assertEqual(row["n_excluded"], 1)
        self.assertEqual(row["n_constant"], 1)
        self.assertEqual(row["n_concordant"], 1)
        self.assertEqual(row["n_discordant"], 1)
        self.assertEqual(row["pairwise_accuracy"], 0.5)
        self.assertEqual(row["avg_correlation"], 0)

    def test_all_ties_or_no_pairs_return_no_estimate(self):
        for data in [two_tier({1: [1, 1], 2: [0, 0]}), two_tier({})]:
            with self.subTest(data=data):
                row = analyse(data, expected_tiers=(0, 1)).row(0, named=True)
                self.assertEqual(row["n_included"], 0)
                self.assertEqual(row["n_concordant"], 0)
                self.assertEqual(row["n_discordant"], 0)
                for name in (
                    "avg_correlation",
                    "pairwise_accuracy",
                    "ci_low",
                    "ci_high",
                ):
                    self.assertIsNone(row[name])
                self.assertEqual(row["ci_status"], "insufficient_samples")

    def test_missing_tier_is_never_reinterpreted_as_a_two_tier_design(self):
        data = two_tier({1: [0, 1], 2: [1, 0]})
        row = analyse(data, expected_tiers=(0, 1, 2)).row(0, named=True)
        self.assertEqual(row["analysis_kind"], "mean_spearman")
        self.assertEqual(row["n_incomplete_tiers"], 2)
        self.assertEqual(row["n_included"], 0)
        self.assertIsNone(row["avg_correlation"])
        self.assertIsNone(row["pairwise_accuracy"])
        with self.assertRaisesRegex(ValueError, "expected integer tier"):
            analyse(complete({1: [0, 1, 2]}), expected_tiers=(0, 1))

    def test_pairwise_exclusion_counts_reconcile(self):
        logs = {
            record(0): log(
                [
                    sample("ok", 0, 0),
                    sample("tie", 0, 1),
                    sample("invalid", 0, 1),
                    sample("missing-score", 0, 0),
                    sample("missing-tier", 0, 0),
                ]
            ),
            record(1): log(
                [
                    sample("ok", 1, 1),
                    sample("tie", 1, 1),
                    sample("invalid", 1, math.inf),
                    sample("missing-score", 1, 1, scores={}),
                ]
            ),
        }
        row = analyse(project(logs), expected_tiers=(0, 1)).row(0, named=True)
        self.assertEqual(row["n_samples"], 5)
        self.assertEqual(row["n_included"], 1)
        self.assertEqual(row["n_excluded"], 4)
        for key in (
            "n_constant",
            "n_invalid_scores",
            "n_missing_scores",
            "n_incomplete_tiers",
        ):
            self.assertEqual(row[key], 1)
        self.assertEqual(row["n_concordant"] + row["n_discordant"], row["n_included"])

    def test_pairwise_validation_requires_review_of_both_outputs(self):
        data = two_tier({1: [0, 1]})
        row = analyse(
            data, expected_tiers=(0, 1), tier_validation_field="reviewed"
        ).row(0, named=True)
        self.assertEqual(row["n_unvalidated"], 1)
        self.assertIsNone(row["pairwise_accuracy"])
        for entry in data.get_logs(type="evaluation").values():
            for observation in entry.samples or []:
                observation.metadata["reviewed"] = True
        row = analyse(
            data, expected_tiers=(0, 1), tier_validation_field="reviewed"
        ).row(0, named=True)
        self.assertEqual(row["pairwise_accuracy"], 1)

    def test_pairwise_interval_is_independent_of_order_and_bootstrap_settings(self):
        data = two_tier({1: [0, 1], 2: [1, 0], 3: [0, 1]})
        first = analyse(data, expected_tiers=(0, 1))
        entries = {
            k: v.model_copy(update={"samples": list(reversed(v.samples or []))})
            for k, v in reversed(list(data.get_logs(type="evaluation").items()))
        }
        second = analyse(
            project(entries), expected_tiers=(1, 0), seed=42, n_resamples=1000
        )
        self.assertTrue(first.equals(second))

    def test_three_tier_results_keep_the_existing_bootstrap(self):
        row = analyse(complete({1: [0, 1, 2], 2: [2, 1, 0]})).row(0, named=True)
        self.assertEqual(row["analysis_kind"], "mean_spearman")
        self.assertEqual(row["ci_status"], "percentile_bootstrap")
        for name in ("n_concordant", "n_discordant", "pairwise_accuracy"):
            self.assertIsNone(row[name])
        self.assertEqual(row["n_resamples"], 2000)
        self.assertEqual(row["seed"], 0)

    def test_pairwise_read_only_log_roundtrip(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp)),
        ):
            stored = Project("pairwise")
            stored.evaluation_log_path.mkdir(parents=True)
            for t in (0, 1):
                path = stored.evaluation_log_path / f"2026_task_{t}.json"
                write_eval_log(
                    log([sample(1, t, t), sample(2, t, 1 - t), sample(3, t, 2)]),
                    location=str(path),
                )
                stored.update_record(
                    record(t), ResultRecord(status="success", log_location=str(path))
                )
            before = {
                str(p): p.read_bytes() for p in Path(tmp).rglob("*") if p.is_file()
            }
            row = analyse(Project("pairwise"), expected_tiers=(0, 1)).row(0, named=True)
            self.assertEqual(row["n_constant"], 1)
            self.assertEqual(row["pairwise_accuracy"], 0.5)
            after = {
                str(p): p.read_bytes() for p in Path(tmp).rglob("*") if p.is_file()
            }
            self.assertEqual(before, after)

    def test_pairwise_uses_comparisons_without_subtraction_overflow(self):
        row = analyse(
            two_tier({1: [-1e308, 1e308], 2: [1e308, -1e308]}), expected_tiers=(0, 1)
        ).row(0, named=True)
        self.assertEqual(row["n_concordant"], 1)
        self.assertEqual(row["n_discordant"], 1)
        self.assertEqual(row["avg_correlation"], 0)


if __name__ == "__main__":
    unittest.main()
