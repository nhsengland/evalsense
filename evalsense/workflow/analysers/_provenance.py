"""Keep identically labelled metrics from different implementations separate."""

from inspect_ai.log import EvalLog

from evalsense.evaluation import EvaluationRecord


def check_metric_implementations(logs: dict[EvaluationRecord, EvalLog]) -> None:
    """Reject mixed implementation versions before grouping or pivoting results.

    A legacy-only project remains readable. Unknown provenance cannot be pooled
    with a recorded implementation under the same evaluator or score label.
    """
    seen: dict[tuple[str, str], str] = {}
    for record, log in logs.items():
        labels = {("evaluator", record.evaluator_name)}
        if log.results:
            labels.update(("score", score.name) for score in log.results.scores)
        for sample in log.samples or []:
            for name, score in (sample.scores or {}).items():
                labels.add(("score", name))
                if isinstance(score.value, dict):
                    labels.update(("score", inner) for inner in score.value)
        for label in labels:
            previous = seen.setdefault(label, record.evaluator_provenance_json)
            if previous != record.evaluator_provenance_json:
                raise ValueError(
                    "Results with the same evaluator or score label have different "
                    "metric implementations (or missing legacy provenance). "
                    "Analyse each implementation separately."
                )
