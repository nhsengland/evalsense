from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """A class for storing the evaluation results."""

    name: str
    category: str
    overall_result: Any
    overall_metadata: dict[str, Any] = field(default_factory=dict)
    instance_results: list[Any] = field(default_factory=list)
    instance_metadata: dict[str, list[Any]] = field(default_factory=dict)
