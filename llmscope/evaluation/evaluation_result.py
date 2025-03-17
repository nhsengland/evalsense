from dataclasses import dataclass, field

type EvaluationValue = (
    str
    | int
    | float
    | bool
    | None
    | list["EvaluationValue"]
    | dict[str, "EvaluationValue"]
)


@dataclass
class EvaluationResult:
    """A class for storing the evaluation results."""

    name: str
    category: str
    overall_result: EvaluationValue
    overall_metadata: dict[str, EvaluationValue] = field(default_factory=dict)
    instance_results: list[EvaluationValue] = field(default_factory=list)
    instance_metadata: dict[str, list[EvaluationValue]] = field(default_factory=dict)
