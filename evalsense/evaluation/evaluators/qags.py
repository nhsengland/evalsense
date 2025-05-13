from abc import abstractmethod
from typing import Any, Literal, Protocol, override

from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import (
    Metric,
    Score,
    Scorer,
    Target,
    mean,
    scorer,
)
from inspect_ai.solver import TaskState

from evalsense.evaluation import (
    Evaluator,
    ScoreCalculator,
    ScorerFactory,
)
from evalsense.generation import ModelConfig
from evalsense.logging import get_logger
from evalsense.utils.text import (
    extract_lines,
    extract_ternary_answer,
    extract_weighted_binary_answer,
)

logger = get_logger(__name__)


class QagsConfig(Protocol):
    """A protocol for configuring QAGS evaluation."""

    answer_comparison_mode: Literal["ternary", "exact", "judge"]
    logprobs: bool
    top_logprobs: int
    ci: float
    debug: bool

    def __init__(
        self,
        answer_comparison_mode: Literal["ternary", "exact", "judge"],
        logprobs: bool = True,
        top_logprobs: int = 20,
        ci: float = 0.1,
        debug: bool = False,
    ):
        """
        Initializes the QAGS configuration.

        Args:
            answer_comparison_mode (Literal["ternary", "exact", "judge"]): The mode
                for comparing answers. Either "ternary", "exact", or "judge".
                In "ternary" mode, the model is expected to answer the generated
                questions with "yes", "no", or "unknown". In other modes, the model
                may give arbitrary answers, which are either compared in terms
                of exact match or compared by the model itself.
            logprobs (bool): Whether to use logprobs to compute weighted answers. Can only
                be used when `answer_comparison_mode` is set to "judge".
            top_logprobs (int): The number of top log probabilities to consider
                when computing weighted answers.
            ci (float): The range near the extreme values (0.0 or 1.0) in which
                to consider the model answer as confident when comparing answers.
                This only affects the score explanation when `answer_comparison_mode`
                is set to "judge". The default value is 0.1, which means that
                answers with a score of 0.9 or are confident "yes", while answers
                with a score of 0.1 or lower are confident "no".
            debug (bool): Whether to report repeated errors in the log.
        """
        self.answer_comparison_mode = answer_comparison_mode
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.ci = ci
        self.debug = debug

    def enforce_not_none[T](self, param_name: str, param_value: T | None) -> T:
        """
        Helper method to enforce that a parameter is not None.

        Args:
            param_name (str): The name of the parameter.
            param_value (T | None): The value of the parameter.

        Raises:
            ValueError: If the parameter value is None.

        Returns:
            T: The parameter value if it is not None.
        """
        if param_value is None:
            raise ValueError(f"{param_name} cannot be None.")
        return param_value

    @abstractmethod
    def get_question_generation_prompt(
        self,
        *,
        source: Literal["prediction", "reference"],
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Constructs the prompt for generating the questions for the model output.

        The prompt should instruct the model to generate each question on
        a separate line.

        Args:
            source (Literal["prediction", "reference"]): The source to use for
                generating the questions. Either "prediction" or "reference".
                According to the source, the generated prompt should either use
                the model output or the reference output/input. When
                `answer_comparison_mode` is set to "ternary", the generated
                questions should be answerable with "yes", "no", or "unknown".
            prediction (str, optional): The model output to evaluate.
            input (str, optional): The input to the model. Optional.
            reference (str, optional): The reference output to compare against.
                Optional.
            metadata (dict[str, Any], optional): Additional Inspect AI sample/task
                state metadata. Optional.

        Returns:
            str: The generated prompt.
        """
        ...

    @abstractmethod
    def get_answer_generation_prompt(
        self,
        *,
        source: Literal["prediction", "reference"],
        question: str,
        prediction: str | None = None,
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Constructs the prompt for generating the answer to a single question.

        Args:
            source (Literal["prediction", "reference"]): The source to use for
                generating the answer. Either "prediction" or "reference".
                According to the source, the generated prompt should either use
                the model output or the reference output/input when asking
                the model to answer the question. When `answer_comparison_mode`
                is set to "ternary", the prompt should instruct the model to
                answer the question with "yes", "no", or "unknown". Otherwise,
                the model should be instructed to give an answer only without
                any further comments.
            prediction (str, optional): The model output to evaluate.
            input (str, optional): The input to the model. Optional.
            reference (str, optional): The reference output to compare against.
                Optional.
            metadata (dict[str, Any], optional): Additional Inspect AI sample/task
                state metadata. Optional.

        Returns:
            str: The generated prompt.
        """
        ...

    def get_answer_comparison_prompt(
        self,
        *,
        question: str,
        prediction_answer: str,
        reference_answer: str,
        input: str | None = None,
        prediction: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Constructs the prompt for comparing answers to the generated questions.

        This method is only used when `answer_comparison_mode` is set to "judge".

        Args:
            question (str): The question to compare answers for.
            prediction_answer (str): The answer generated from the model output.
            reference_answer (str): The answer generated from the reference output.
            input (str | None, optional): The input to the model. Optional.
            prediction (str | None, optional): The model output to evaluate. Optional.
            reference (str | None, optional): The reference output to compare against.
                Optional.
            metadata (dict[str, Any] | None, optional): Additional Inspect AI sample/task
                state metadata. Optional.

        Returns:
            str: The generated prompt.
        """
        if self.answer_comparison_mode == "judge":
            raise NotImplementedError(
                "Answer comparison prompt generation is not implemented. "
                "If you want to use QAGS in judge mode, please implement this method."
            )
        assert False, (
            "Should not attempt to generate comparison prompt in non-judge mode."
        )


class QagsScoreCalculator(ScoreCalculator):
    """QAGS score calculator."""

    _symbol_dict: dict[bool | None, str] = {
        True: "✅",
        False: "❌",
        None: "❓",
    }

    def __init__(
        self,
        model: Model,
        config: QagsConfig,
        name: str = "QAGS",
        debug: bool = False,
    ):
        """
        Initializes the QAGS score calculator.

        Args:
            model (Model): The model to use for evaluation.
            config (QagsConfig): The configuration for the QAGS score calculator.
            name (str): The name of the score calculator. Defaults to "QAGS".
            debug (bool): Whether to report repeated errors in the log.
        """
        self.model = model
        self.config = config
        self.name = name
        self.warned_weighted_answer = False

    @property
    def generate_config(self) -> GenerateConfig:
        """Generation configuration for the model."""
        if self.config.logprobs and self.config.answer_comparison_mode == "judge":
            return GenerateConfig(
                logprobs=self.config.logprobs,
                top_logprobs=self.config.top_logprobs,
            )
        return GenerateConfig()

    @override
    def calculate(
        self,
        *,
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: dict,
    ) -> Score:
        """This method is not supported for QAGS and will raise an error when called.

        Use `calculate_async` instead.

        Args:
            prediction (str): The model output to evaluate.
            input (str, optional): The input to the model. Optional.
            reference (str, optional): The reference output to compare against.
                Optional.
            metadata (dict[str, Any], optional): Additional Inspect AI sample/task
                state metadata. Optional.
            **kwargs (dict): Additional keyword arguments specific to the given
                evaluation method.

        Returns:
            Score: The Inspect AI Score object with the calculated result.

        Raises:
            NotImplementedError: When called, as synchronous evaluation is not
                supported for QAGS.
        """
        raise NotImplementedError(
            "Synchronous evaluation is not supported for QAGS. "
            "Use calculate_async instead."
        )

    async def _generate_questions(
        self,
        *,
        prediction: str,
        score_metadata: dict[str, Any],
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Generates questions for the model output and reference output.

        Args:
            prediction (str): The model output to evaluate.
            score_metadata (dict[str, Any]): The dictionary for storing metadata
                associated with the evaluation, returned with the score.
            input (str | None, optional): The input to the model. Optional.
            reference (str | None, optional): The reference output to compare against.
                Optional.
            metadata (dict[str, Any] | None, optional): Additional Inspect AI sample/task
                state metadata. Optional.

        Returns:
            list[str]: A list of generated questions.
        """
        # Questions for model outputs
        prediction_questions_prompt = self.config.get_question_generation_prompt(
            source="prediction",
            prediction=prediction,
            input=input,
            reference=reference,
            metadata=metadata,
        )
        # We don't actually need the logprobs until comparing the answers,
        # but the vLLM provider uses the config from the first sample in the batch
        # so we need to use consistent config for all samples.
        prediction_questions_output = await self.model.generate(
            prediction_questions_prompt, config=self.generate_config
        )
        prediction_questions = extract_lines(
            prediction_questions_output.completion,
            include_filter_fun=lambda line: line.endswith("?"),
        )

        # Questions for reference outputs
        reference_questions_prompt = self.config.get_question_generation_prompt(
            source="reference",
            prediction=prediction,
            input=input,
            reference=reference,
            metadata=metadata,
        )
        reference_questions_output = await self.model.generate(
            reference_questions_prompt, config=self.generate_config
        )
        reference_questions = extract_lines(
            reference_questions_output.completion,
            include_filter_fun=lambda line: line.endswith("?"),
        )

        questions = prediction_questions + reference_questions

        score_metadata["questions"] = questions
        score_metadata["prediction_questions_prompt"] = prediction_questions_prompt
        score_metadata["reference_questions_prompt"] = reference_questions_prompt
        score_metadata["raw_prediction_questions"] = (
            prediction_questions_output.completion
        )
        score_metadata["raw_reference_questions"] = (
            reference_questions_output.completion
        )

        return questions

    async def _generate_answers(
        self,
        *,
        prediction: str,
        score_metadata: dict[str, Any],
        questions: list[str],
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Generates answers for the model output and reference output.

        Args:
            prediction (str): The model output to evaluate.
            score_metadata (dict[str, Any]): The dictionary for storing metadata
                associated with the evaluation, returned with the score.
            questions (list[str]): The list of questions to generate answers for.
            input (str | None, optional): The input to the model. Optional.
            reference (str | None, optional): The reference output to compare against.
                Optional.
            metadata (dict[str, Any] | None, optional): Additional Inspect AI sample/task
                state metadata. Optional.

        Returns:
            tuple[list[str], list[str]]: A tuple containing two lists of generated
                answers - one for the model output and one for the reference output,
                respectively.
        """
        prediction_answers: list[str] = []
        reference_answers: list[str] = []
        score_metadata["raw_prediction_answers"] = []
        score_metadata["raw_reference_answers"] = []
        score_metadata["prediction_answer_prompts"] = []
        score_metadata["reference_answer_prompts"] = []
        for question in questions:
            prediction_answer_prompt = self.config.get_answer_generation_prompt(
                source="prediction",
                question=question,
                prediction=prediction,
                input=input,
                reference=reference,
                metadata=metadata,
            )
            prediction_answer_output = await self.model.generate(
                prediction_answer_prompt, config=self.generate_config
            )
            prediction_answers.append(prediction_answer_output.completion)

            reference_answer_prompt = self.config.get_answer_generation_prompt(
                source="reference",
                question=question,
                prediction=prediction,
                input=input,
                reference=reference,
                metadata=metadata,
            )
            reference_answer_output = await self.model.generate(
                reference_answer_prompt, config=self.generate_config
            )
            reference_answers.append(reference_answer_output.completion)

            score_metadata["raw_prediction_answers"].append(
                prediction_answer_output.completion
            )
            score_metadata["raw_reference_answers"].append(
                reference_answer_output.completion
            )
            score_metadata["prediction_answer_prompts"].append(prediction_answer_prompt)
            score_metadata["reference_answer_prompts"].append(reference_answer_prompt)

        return prediction_answers, reference_answers

    def _evaluate_ternary_answers(
        self,
        *,
        prediction: str,
        questions: list[str],
        raw_prediction_answers: list[str],
        raw_reference_answers: list[str],
        score_metadata: dict[str, Any],
    ) -> Score:
        """Evaluates the answers using the ternary answer comparison mode.

        Args:
            prediction (str): The model output to evaluate.
            questions (list[str]): The list of questions generated for the model
                output.
            raw_prediction_answers (list[str]): The list of answers generated from
                the model output.
            raw_reference_answers (list[str]): The list of answers generated from
                the reference output.
            score_metadata (dict[str, Any]): The dictionary for storing metadata
                associated with the evaluation, returned with the score.

        Returns:
            Score: The Inspect AI Score object with the calculated result.
        """
        prediction_answers = [
            extract_ternary_answer(answer, binary_only=False, unknown_on_mismatch=True)
            for answer in raw_prediction_answers
        ]
        reference_answers = [
            extract_ternary_answer(answer, binary_only=False, unknown_on_mismatch=True)
            for answer in raw_reference_answers
        ]

        ref_positive = sum([ra is True for ra in reference_answers])
        pred_positive = sum([pa is True for pa in prediction_answers])
        true_positive = sum(
            [
                pa == ra and ra is True
                for pa, ra in zip(prediction_answers, reference_answers)
            ]
        )
        total_correct = sum(
            [pa == ra for pa, ra in zip(prediction_answers, reference_answers)]
        )

        coverage = true_positive / ref_positive if ref_positive > 0 else 0.0
        groundedness = true_positive / pred_positive if pred_positive > 0 else 0.0
        accuracy = (
            total_correct / len(prediction_answers)
            if len(prediction_answers) > 0
            else 0.0
        )

        explanation = "QAGS Evaluation Report\n\n\nMismatched Q&As\n"
        for i, (question, pa, ra) in enumerate(
            zip(questions, prediction_answers, reference_answers)
        ):
            if pa == ra:
                continue
            explanation += (
                f"* [{i}] Q: {question}, PA: {self._symbol_dict.get(pa)}, "
                f"RA: {self._symbol_dict.get(ra)}, "
                f"Match: {self._symbol_dict.get(False)}\n"
            )
        explanation += "\n\nAll Q&As\n"
        for i, (question, pa, ra) in enumerate(
            zip(questions, prediction_answers, reference_answers)
        ):
            explanation += (
                f"* [{i}] Q: {question}, PA: {self._symbol_dict.get(pa)}, "
                f"RA: {self._symbol_dict.get(ra)}, "
                f"Match: {self._symbol_dict.get(pa == ra)}\n"
            )
        explanation += (
            "\n\n"
            + f"Coverage: {coverage:.2f} ({true_positive}/{ref_positive})\n"
            + f"Groundedness: {groundedness:.2f} ({true_positive}/{pred_positive})\n"
            + f"Accuracy: {accuracy:.2f} ({total_correct}/{len(prediction_answers)})"
        )

        score_metadata["prediction_answers"] = prediction_answers
        score_metadata["reference_answers"] = reference_answers
        score_metadata = {"explanation": explanation} | score_metadata

        return Score(
            value={
                f"{self.name} Coverage": coverage,
                f"{self.name} Groundedness": groundedness,
                f"{self.name} Accuracy": accuracy,
            },
            answer=prediction,
            explanation=explanation,
            metadata=score_metadata,
        )

    def _evaluate_exact_answers(
        self,
        *,
        prediction: str,
        questions: list[str],
        raw_prediction_answers: list[str],
        raw_reference_answers: list[str],
        score_metadata: dict[str, Any],
    ) -> Score:
        """Evaluates the answers using the exact answer comparison mode.

        Args:
            prediction (str): The model output to evaluate.
            questions (list[str]): The list of questions generated for the model
                output.
            raw_prediction_answers (list[str]): The list of answers generated from
                the model output.
            raw_reference_answers (list[str]): The list of answers generated from
                the reference output.
            score_metadata (dict[str, Any]): The dictionary for storing metadata
                associated with the evaluation, returned with the score.

        Returns:
            Score: The Inspect AI Score object with the calculated result.
        """
        prediction_answers = [pa.strip().lower() for pa in raw_prediction_answers]
        reference_answers = [ra.strip().lower() for ra in raw_reference_answers]
        total_correct = sum(
            [pa == ra for pa, ra in zip(prediction_answers, reference_answers)]
        )
        accuracy = (
            total_correct / len(prediction_answers)
            if len(prediction_answers) > 0
            else 0.0
        )

        explanation = "QAGS Evaluation Report\n\n\nMismatched Q&As\n"
        for i, (question, pa, ra) in enumerate(
            zip(questions, prediction_answers, reference_answers)
        ):
            if pa == ra:
                continue
            explanation += (
                f"* [{i}] Q: {question}, PA: {pa}, RA: {ra}, "
                f"Match: {self._symbol_dict.get(False)}\n"
            )
        explanation += "\n\nAll Q&As\n"
        for i, (question, pa, ra) in enumerate(
            zip(questions, prediction_answers, reference_answers)
        ):
            explanation += (
                f"* [{i}] Q: {question}, PA: {pa}, RA: {ra},"
                f"Match: {self._symbol_dict.get(pa == ra)}\n"
            )
        explanation += (
            f"\n\nAccuracy: {accuracy:.2f} ({total_correct}/{len(prediction_answers)})"
        )

        score_metadata["prediction_answers"] = prediction_answers
        score_metadata["reference_answers"] = reference_answers
        score_metadata = {"explanation": explanation} | score_metadata

        return Score(
            value=accuracy,
            answer=prediction,
            explanation=explanation,
            metadata=score_metadata,
        )

    async def _evaluate_judge_answers(
        self,
        prediction: str,
        questions: list[str],
        raw_prediction_answers: list[str],
        raw_reference_answers: list[str],
        score_metadata: dict[str, Any],
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Score:
        """Evaluates the answers using the judge answer comparison mode.

        Args:
            prediction (str): The model output to evaluate.
            questions (list[str]): The list of questions generated for the model
                output.
            raw_prediction_answers (list[str]): The list of answers generated from
                the model output.
            raw_reference_answers (list[str]): The list of answers generated from
                the reference output.
            score_metadata (dict[str, Any]): The dictionary for storing metadata
                associated with the evaluation, returned with the score.
            input (str | None, optional): The input to the model. Optional.
            reference (str | None, optional): The reference output to compare against.
                Optional.
            metadata (dict[str, Any] | None, optional): Additional Inspect AI sample/task
                state metadata. Optional.

        Returns:
            Score: The Inspect AI Score object with the calculated result.
        """
        answer_comparisons: list[float] = []
        for question, prediction_answer, reference_answer in zip(
            questions,
            raw_prediction_answers,
            raw_reference_answers,
        ):
            answer_comparison_prompt = self.config.get_answer_comparison_prompt(
                question=question,
                prediction_answer=prediction_answer,
                reference_answer=reference_answer,
                input=input,
                prediction=prediction,
                reference=reference,
                metadata=metadata,
            )
            answer_comparison_output = await self.model.generate(
                answer_comparison_prompt, config=self.generate_config
            )
            answer_comparison = float(
                extract_ternary_answer(
                    answer_comparison_output.completion,
                    binary_only=True,
                    unknown_on_mismatch=False,
                )
            )
            if self.config.logprobs:
                try:
                    answer_comparison = extract_weighted_binary_answer(
                        answer_comparison_output
                    )
                except ValueError as e:
                    if not self.warned_weighted_answer or self.config.debug:
                        self.warned_weighted_answer = True

                        error_message = (
                            f"❌  Cannot compute weighted comparison score: {e} "
                            "Falling back to binary comparison."
                        )

                        if not self.config.debug:
                            error_message += (
                                " Further errors will be suppressed "
                                + "(set debug=True to see all errors)."
                            )

                        logger.error(error_message)
            answer_comparisons.append(answer_comparison)

        def to_match_symbol(answer_comparison: float) -> str:
            if answer_comparison > 1 - self.config.ci:
                return self._symbol_dict[True]
            elif answer_comparison < self.config.ci:
                return self._symbol_dict[False]
            else:
                return self._symbol_dict[None]

        accuracy = sum(answer_comparisons) / len(answer_comparisons)

        explanation = "QAGS Evaluation Report\n\n\nMismatched Q&As\n"
        for i, (question, pa, ra, ac) in enumerate(
            zip(
                questions,
                raw_prediction_answers,
                raw_reference_answers,
                answer_comparisons,
            )
        ):
            if ac > 1 - self.config.ci:
                continue
            explanation += (
                f"* [{i}] Q: {question}, PA: {pa}, RA: {ra}, Score: {ac:.2f}, "
                f"Match: {to_match_symbol(ac)}\n"
            )
        explanation += "\n\nAll Q&As\n"
        for i, (question, pa, ra, ac) in enumerate(
            zip(
                questions,
                raw_prediction_answers,
                raw_reference_answers,
                answer_comparisons,
            )
        ):
            explanation += (
                f"* [{i}] Q: {question}, PA: {pa}, RA: {ra}, Score: {ac:.2f}, "
                f"Match: {to_match_symbol(ac)}\n"
            )
        explanation += (
            f"\n\nAccuracy: {accuracy:.2f} "
            + f"({sum(answer_comparisons):.2f}/{len(answer_comparisons)})"
        )

        score_metadata["prediction_answers"] = raw_prediction_answers
        score_metadata["reference_answers"] = raw_reference_answers
        score_metadata["answer_comparisons"] = answer_comparisons
        score_metadata = {"explanation": explanation} | score_metadata

        return Score(
            value=accuracy,
            answer=prediction,
            explanation=explanation,
            metadata=score_metadata,
        )

    @override
    async def calculate_async(
        self,
        *,
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: dict,
    ) -> Score:
        """Asynchronously computes evaluation scores for QAGS.

        Args:
            prediction (str): The model output to evaluate.
            input (str, optional): The input to the model. Optional.
            reference (str, optional): The reference output to compare against.
                Optional.
            metadata (dict[str, Any], optional): Additional Inspect AI sample/task
                state metadata. Optional.
            **kwargs (dict): Additional keyword arguments specific to the given
                evaluation method.

        Returns:
            Score: The Inspect AI Score object with the calculated result.
        """

        score_metadata = {}

        all_questions = await self._generate_questions(
            prediction=prediction,
            score_metadata=score_metadata,
            input=input,
            reference=reference,
            metadata=metadata,
        )

        prediction_answers, reference_answers = await self._generate_answers(
            prediction=prediction,
            score_metadata=score_metadata,
            questions=all_questions,
            input=input,
            reference=reference,
            metadata=metadata,
        )

        match self.config.answer_comparison_mode:
            case "ternary":
                return self._evaluate_ternary_answers(
                    prediction=prediction,
                    questions=all_questions,
                    raw_prediction_answers=prediction_answers,
                    raw_reference_answers=reference_answers,
                    score_metadata=score_metadata,
                )
            case "exact":
                return self._evaluate_exact_answers(
                    prediction=prediction,
                    questions=all_questions,
                    raw_prediction_answers=prediction_answers,
                    raw_reference_answers=reference_answers,
                    score_metadata=score_metadata,
                )
            case "judge":
                return await self._evaluate_judge_answers(
                    prediction=prediction,
                    questions=all_questions,
                    raw_prediction_answers=prediction_answers,
                    raw_reference_answers=reference_answers,
                    score_metadata=score_metadata,
                    input=input,
                    reference=reference,
                    metadata=metadata,
                )
            case _:
                raise ValueError(
                    f"Invalid answer comparison mode: {self.config.answer_comparison_mode}. "
                    "Expected one of 'ternary', 'exact', 'judge'."
                )


class QagsScorerFactory(ScorerFactory):
    """Scorer factory for QAGS."""

    def __init__(
        self,
        name: str,
        config: QagsConfig,
        metrics: list[Metric | dict[str, list[Metric]]]
        | dict[str, list[Metric]]
        | None = None,
    ):
        """
        Initialize the QAGS scorer factory.

        Args:
            config (QagsConfig): The configuration for the QAGS scorer.
            metrics (list[Metric | dict[str, list[Metric]]] | dict[str, list[Metric]] | None):
                The metrics to use for the evaluation. If `None`, the default metric
                will be used (G-Eval).
        """
        self.name = name
        self.config = config
        if metrics is None:
            if self.config.answer_comparison_mode == "ternary":
                metrics = [
                    {
                        f"{name} Coverage": [mean()],
                        f"{name} Groundedness": [mean()],
                        f"{name} Accuracy": [mean()],
                    }
                ]
            else:
                metrics = [mean()]
        self.metrics = metrics

    @override
    def create_scorer(self, model: Model) -> Scorer:
        """
        Creates a QAGS scorer.

        Args:
            model (Model): The model to create a scorer for.

        Returns:
            Scorer: The created QAGS scorer.
        """

        @scorer(name=self.name, metrics=self.metrics)
        def qags_scorer() -> Scorer:
            qags_score_calculator = QagsScoreCalculator(
                model=model,
                config=self.config,
                name=self.name,
            )

            async def score(state: TaskState, target: Target):
                return await qags_score_calculator.calculate_async(
                    input=state.input_text,
                    prediction=state.output.completion,
                    reference=target.text,
                    metadata=state.metadata,
                )

            return score

        return qags_scorer()


def get_qags_evaluator(
    *,
    config: QagsConfig,
    name: str = "QAGS",
    model_name: str | None = None,
    metrics: list[Metric | dict[str, list[Metric]]]
    | dict[str, list[Metric]]
    | None = None,
    model_config: ModelConfig,
) -> Evaluator:
    """
    Constructs a QAGS evaluator that can be used in EvalSense evaluation pipeline.

    Args:
        config (QagsConfig): The configuration for the QAGS evaluator.
        name (str): The name of the QAGS evaluator.
        model_name (str | None): The name of the model to use for evaluation.
            If `None`, the name from the model configuration will be used.
        metrics (list[Metric | dict[str, list[Metric]]] | dict[str, list[Metric]] | None):
            The metrics to use for the evaluation. If `None`, the default metrics
            will be used (QAGS precision, recall and F1).
        model_config (ModelConfig): The configuration of the model to be used
            for evaluation.

    Returns:
        Evaluator: The constructed QAGS evaluator.
    """
    metric_name = f"{name} ({model_name or model_config.name})"
    return Evaluator(
        name=metric_name,
        scorer=QagsScorerFactory(
            name=metric_name,
            config=config,
            metrics=metrics,
        ),
        model_config=model_config,
    )
