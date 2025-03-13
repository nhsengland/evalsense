import evaluate
from typing import override

from llmscope.evaluation import Evaluator, EvaluationResult, ResultCategory


class BleuEvaluator(Evaluator):
    """A class for evaluating LLM outputs using BLEU score."""

    def __init__(self) -> None:
        """Initializes the evaluator with a name and optional category."""
        super().__init__("bleu")

    @override
    def _evaluate_outputs(
        self,
        outputs: list[str],
        references: list[str] | list[list[str]] | None = None,
        **kwargs,
    ) -> list[EvaluationResult]:
        """Evaluates the LLM outputs using BLEU score.

        Args:
            outputs (list[str]): The LLM outputs to evaluate.
            references (list[str] | None): The reference values, if available.
            **kwargs: Additional keyword arguments.

        Returns:
            list[EvaluationResult]: The evaluation result.
        """
        if references is None:
            raise ValueError("References are required for BLEU evaluation.")
        if references and type(references[0]) is str:
            references = [[r] for r in references]  # type: ignore[assignment]

        bleu = evaluate.load("bleu")
        result = bleu.compute(predictions=outputs, references=references)

        if result is None:
            raise ValueError("Unexpected None result from BLEU evaluation.")

        return [
            EvaluationResult(
                name=self.name,
                category=ResultCategory.STATISTICAL.value,
                overall_result=result["bleu"],
                overall_metadata={
                    "brevity_penalty": result["brevity_penalty"],
                    "length_ratio": result["length_ratio"],
                    "translation_length": result["translation_length"],
                    "reference_length": result["reference_length"],
                },
                instance_metadata={
                    "precisions": result["precisions"],
                },
            )
        ]
