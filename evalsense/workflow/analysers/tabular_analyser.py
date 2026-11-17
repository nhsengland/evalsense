from collections import Counter
import hashlib
import json
from typing import Literal, cast, override

import pandas as pd
import polars as pl

from inspect_ai.log import EvalMetric, EvalScore

from evalsense.evaluation import EvaluationRecord
from evalsense.workflow import Project, ResultAnalyser

from evalsense.workflow.analysers._provenance import check_metric_implementations

OUTPUT_FORMATTERS = {
    "polars": lambda df: df,
    "pandas": lambda df: df.to_pandas(),
}


def _metric_id(
    record: EvaluationRecord, score: EvalScore, key: str, metric: EvalMetric
) -> str:
    """Identify the recorded evaluator and aggregate independently of their labels."""
    identity = [
        record.evaluator_name,
        record.evaluator_fingerprint,
        record.evaluator_provenance_json,
        score.model_dump(mode="json", include={"name", "scorer", "reducer", "params"}),
        key,
        metric.model_dump(mode="json", include={"name", "params"}),
    ]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _metric_labels(columns: dict[str, tuple[str, str]]) -> dict[str, str]:
    counts = Counter(label for label, _ in columns.values())
    labels = {
        key: label if counts[label] == 1 else f"{label} [{evaluator}, {key[:12]}]"
        for key, (label, evaluator) in columns.items()
    }
    if len(set(labels.values())) != len(labels):
        raise ValueError("Ambiguous summary column names after qualifying metrics.")
    return labels


class TabularResultAnalyser[T: pl.DataFrame | pd.DataFrame](ResultAnalyser[T]):
    """An analyser summarising evaluation results in a tabular format.

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
        check_metric_implementations(eval_logs)

        result_data = []
        columns: dict[str, tuple[str, str]] = {}
        seen: set[tuple[str, str]] = set()
        for eval_record, log in eval_logs.items():
            if not log.results:
                continue

            generation_id = eval_record.generation_id
            for score in log.results.scores:
                for metric_name, metric in score.metrics.items():
                    label = f"{score.name}/{metric_name}"
                    key = _metric_id(eval_record, score, metric_name, metric)
                    columns[key] = (label, eval_record.evaluator_name)
                    cell = (generation_id, key)
                    if cell in seen:
                        raise ValueError(
                            f"Duplicate summary metric {label!r} for generation {generation_id}."
                        )
                    seen.add(cell)

                    result_data.append(
                        {
                            "dataset": eval_record.dataset_record.name,
                            "splits": ", ".join(eval_record.dataset_record.splits),
                            "task": eval_record.task_name,
                            "generator": eval_record.generator_name,
                            "model": eval_record.model_record.name,
                            "generation_id": generation_id,
                            "dataset_version": eval_record.dataset_record.version,
                            "experiment_name": eval_record.experiment_name,
                            "metric": key,
                            "value": metric.value,
                        }
                    )

        df = pl.DataFrame(result_data, schema_overrides={"experiment_name": pl.String})
        df = df.pivot(
            on="metric",
            index=[
                "dataset",
                "splits",
                "task",
                "generator",
                "model",
                "generation_id",
                "dataset_version",
                "experiment_name",
            ],
            values="value",
        ).rename(_metric_labels(columns))
        if self.output_format in OUTPUT_FORMATTERS:
            return cast(T, OUTPUT_FORMATTERS[self.output_format](df))
        raise ValueError(
            f"Invalid output format: {self.output_format}. "
            f"Must be one of: {', '.join(OUTPUT_FORMATTERS.keys())}."
        )
