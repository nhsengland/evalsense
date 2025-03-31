from dataclasses import dataclass

from inspect_ai.solver import Solver


@dataclass
class GenerationSteps:
    """A class for specifying generation steps for LLMs, including prompting."""

    name: str
    solver: Solver | list[Solver]
