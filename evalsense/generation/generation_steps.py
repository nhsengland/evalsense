from dataclasses import dataclass

from inspect_ai.solver import Solver

from evalsense.utils.cache import fingerprint


# This class wraps solvers to provide a unique identifier (name) for each
# generation procedure to be used in experiments, allowing the users to
# run experiments with different generation steps (e.g., using different
# prompts or tools) and compare the results.
@dataclass
class GenerationSteps:
    """A class for specifying generation steps for LLMs, including prompting."""

    name: str
    steps: Solver | list[Solver]
    version: str | None = None

    @property
    def cache_key(self) -> str:
        """Identity of solver parameters and the caller's custom-code version."""
        return fingerprint(self.steps, version=self.version)
