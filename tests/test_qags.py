import itertools
import json
from pathlib import Path
import tempfile
from typing import Any, Literal, cast
import unittest
from unittest.mock import AsyncMock, Mock, patch

from datasets import Dataset, DatasetDict
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import Model, ModelName, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState, generate

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import Evaluator, ExperimentConfig
from evalsense.evaluation.evaluators.qags import (
    QagsConfig,
    QagsScoreCalculator,
    QagsScorerFactory,
    get_qags_evaluator,
)
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.workflow import Pipeline, Project


class FixtureConfig(QagsConfig):
    def get_question_generation_prompt(
        self,
        *,
        source: Literal["prediction", "reference"],
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return json.dumps(
            dict(
                stage="questions",
                source=source,
                prediction=prediction,
                input=input,
                reference=reference,
                metadata=metadata,
            )
        )

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
        return json.dumps(
            dict(
                stage="answer",
                source=source,
                question=question,
                prediction=prediction,
                input=input,
                reference=reference,
                metadata=metadata,
            )
        )

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
        return json.dumps(
            dict(
                stage="comparison",
                question=question,
                prediction_answer=prediction_answer,
                reference_answer=reference_answer,
                input=input,
                prediction=prediction,
                reference=reference,
                metadata=metadata,
            )
        )


def model_with(*completions: str) -> Model:
    model = Mock(spec=Model)
    model.generate = AsyncMock(
        side_effect=[
            ModelOutput.from_content(model="fixture", content=text)
            for text in completions
        ]
    )
    return cast(Model, model)


def calculator(
    mode: Literal["ternary", "exact", "judge"] = "ternary",
    model: Model | None = None,
    logprobs: bool = False,
) -> QagsScoreCalculator:
    return QagsScoreCalculator(
        model or model_with(), FixtureConfig(mode, logprobs=logprobs), name="fixture"
    )


def ternary(
    prediction: list[str], reference: list[str], questions: list[str] | None = None
):
    return calculator()._evaluate_ternary_answers(
        prediction="candidate",
        questions=questions
        if questions is not None
        else [f"Question {i}?" for i in range(len(prediction))],
        raw_prediction_answers=prediction,
        raw_reference_answers=reference,
        score_metadata={},
    )


class AnswerValidationTests(unittest.TestCase):
    def test_two_malformed_answers_cannot_match(self):
        with self.assertRaises(ValueError):
            ternary(["unparseable"], ["unparseable"])

    def test_malformed_answers_fail_on_either_side(self):
        invalid = [
            "",
            " ",
            "maybe",
            "yes or no",
            "no response received",
            "not true",
            "Yes. No.",
            "I do not know",
            "I don't know if yes",
            "Unknown cause",
            'The answer is "yes"',
            '{"answer": "yes"}',
            "false positive",
            "No evidence in this note",
            "Here are the options: yes, no, unknown",
        ]
        for text, side in itertools.product(invalid, ("prediction", "reference")):
            with self.subTest(text=text, side=side), self.assertRaises(ValueError):
                ternary(
                    [text if side == "prediction" else "yes"],
                    [text if side == "reference" else "yes"],
                )

    def test_full_answers_allow_documented_variants(self):
        for variants, expected in [
            (["yes", " YES. ", "True!", "true"], True),
            (["no", "NO.", "false!", "false"], False),
            (["unknown", "UNKNOWN.", "I don't know", "I don’t know!"], None),
        ]:
            for text in variants:
                with self.subTest(text=text):
                    result = ternary([text], [text])
                    assert result.metadata is not None
                    self.assertIs(result.metadata["prediction_answers"][0], expected)
                    assert isinstance(result.value, dict)
                    self.assertEqual(result.value["fixture Accuracy"], 1.0)

    def test_unknown_is_distinct_from_invalid(self):
        score = ternary(["unknown"], ["I don't know"])
        self.assertEqual(
            score.value,
            {
                "fixture Coverage": 0.0,
                "fixture Groundedness": 0.0,
                "fixture Accuracy": 1.0,
            },
        )
        with self.assertRaises(ValueError):
            ternary(["unknown"], ["unparseable"])

    def test_ternary_truth_table_preserves_defined_scores(self):
        # The positive-only coverage/groundedness semantics belong to issue #57.
        # This change must not silently redefine those quantities.
        for pa, ra in itertools.product(("yes", "no", "unknown"), repeat=2):
            with self.subTest(prediction=pa, reference=ra):
                score = ternary([pa], [ra])
                positive = float(pa == ra == "yes")
                self.assertEqual(
                    score.value,
                    {
                        "fixture Coverage": positive,
                        "fixture Groundedness": positive,
                        "fixture Accuracy": float(pa == ra),
                    },
                )

    def test_mixed_valid_answers_match_hand_calculation(self):
        score = ternary(["no", "yes", "yes", "yes"], ["yes", "no", "yes", "unknown"])
        self.assertEqual(
            score.value,
            {
                "fixture Coverage": 0.5,
                "fixture Groundedness": 1 / 3,
                "fixture Accuracy": 0.25,
            },
        )

    def test_invalid_answer_does_not_change_the_denominator(self):
        with self.assertRaises(ValueError):
            ternary(["yes", "unparseable"], ["yes", "unparseable"])

    def test_empty_and_misaligned_batches_fail(self):
        for questions, pa, ra in [
            ([], [], []),
            (["One?"], [], ["yes"]),
            (["One?"], ["yes"], []),
            (["One?"], ["yes", "no"], ["yes"]),
            (["One?", "Two?"], ["yes"], ["yes"]),
        ]:
            for mode in ("ternary", "exact"):
                with (
                    self.subTest(mode=mode, lengths=(len(questions), len(pa), len(ra))),
                    self.assertRaises(ValueError),
                ):
                    evaluator = calculator()
                    method = (
                        evaluator._evaluate_ternary_answers
                        if mode == "ternary"
                        else evaluator._evaluate_exact_answers
                    )
                    method(
                        prediction="candidate",
                        questions=questions,
                        raw_prediction_answers=pa,
                        raw_reference_answers=ra,
                        score_metadata={},
                    )

    def test_empty_exact_answers_cannot_match(self):
        for pa, ra in [("", ""), (" ", "\n"), ("answer", ""), ("", "answer")]:
            with (
                self.subTest(prediction=pa, reference=ra),
                self.assertRaises(ValueError),
            ):
                calculator("exact")._evaluate_exact_answers(
                    prediction="candidate",
                    questions=["One?"],
                    raw_prediction_answers=[pa],
                    raw_reference_answers=[ra],
                    score_metadata={},
                )

    def test_exact_mode_keeps_nonempty_text_normalisation(self):
        score = calculator("exact")._evaluate_exact_answers(
            prediction="candidate",
            questions=["One?", "Two?"],
            raw_prediction_answers=[" London ", "Paris"],
            raw_reference_answers=["LONDON", "Rome"],
            score_metadata={},
        )
        self.assertEqual(score.value, 0.5)

    def test_error_does_not_echo_source_text(self):
        with self.assertRaisesRegex(ValueError, "prediction.*question 0") as context:
            ternary(["sensitive patient information"], ["yes"])
        self.assertNotIn("sensitive patient information", str(context.exception))

    def test_validation_error_is_public_and_distinct_from_unknown(self):
        from evalsense.evaluation.evaluators import QagsValidationError

        self.assertTrue(issubclass(QagsValidationError, ValueError))
        with self.assertRaises(QagsValidationError):
            ternary(["malformed"], ["unknown"])

    def test_invalid_mode_is_rejected_before_model_calls(self):
        with self.assertRaises(ValueError):
            calculator(cast(Any, "unsupported"))


class AsyncQagsTests(unittest.IsolatedAsyncioTestCase):
    async def test_question_normalisation_deduplicates_across_both_sources(self):
        prediction = "1. Does it work?\n2) Does it work?\n* What is measured?\n?\nAn introduction\n"
        reference = (
            "- Does it work?   \n[3] What is measured?\n• Is it reliable?\nWhat? Why?"
        )
        fake = model_with(prediction, reference)
        metadata: dict[str, Any] = {}
        questions = await calculator(model=fake)._generate_questions(
            prediction="candidate", reference="reference", score_metadata=metadata
        )
        self.assertEqual(
            questions, ["Does it work?", "What is measured?", "Is it reliable?"]
        )
        self.assertEqual(
            metadata["question_sources"],
            [["prediction", "reference"], ["prediction", "reference"], ["reference"]],
        )
        self.assertEqual(
            metadata["question_filtering"],
            {
                "candidate_lines": 9,
                "invalid_lines": 3,
                "duplicate_questions": 3,
                "retained_questions": 3,
            },
        )
        self.assertEqual(metadata["raw_prediction_questions"], prediction)
        self.assertEqual(metadata["raw_reference_questions"], reference)

    async def test_question_normalisation_preserves_numeric_and_negative_content(self):
        fake = model_with(
            "Is the dose 1.5 mg?\nIs the dose -2 mg?\nIs the dose 2 mg?", ""
        )
        questions = await calculator(model=fake)._generate_questions(
            prediction="candidate", score_metadata={}
        )
        self.assertEqual(
            questions,
            ["Is the dose 1.5 mg?", "Is the dose -2 mg?", "Is the dose 2 mg?"],
        )

    async def test_question_case_and_quoted_whitespace_remain_distinct(self):
        originals = [
            "Is the unit mM?",
            "Is the unit mm?",
            'Does the value equal "a  b"?',
            'Does the value equal "a b"?',
        ]
        fake = model_with("\n".join(originals), "")
        questions = await calculator(model=fake)._generate_questions(
            prediction="candidate", score_metadata={}
        )
        self.assertEqual(questions, originals)

    async def test_unicode_equivalent_questions_are_deduplicated(self):
        fake = model_with("Is café mentioned?", "Is cafe\u0301 mentioned?")
        metadata: dict[str, Any] = {}
        questions = await calculator(model=fake)._generate_questions(
            prediction="candidate", score_metadata=metadata
        )
        self.assertEqual(questions, ["Is café mentioned?"])
        self.assertEqual(metadata["question_filtering"]["duplicate_questions"], 1)

    async def test_empty_question_set_fails_before_answer_generation_in_all_modes(self):
        for mode in ("ternary", "exact", "judge"):
            with self.subTest(mode=mode):
                fake = model_with("No questions available", "?\n---")
                with self.assertRaisesRegex(ValueError, "No valid questions"):
                    await calculator(cast(Any, mode), model=fake).calculate_async(
                        prediction="candidate"
                    )
                self.assertEqual(cast(Any, fake.generate).await_count, 2)

    async def test_duplicates_are_answered_once_and_cannot_inflate_agreement(self):
        outcomes = []
        for text in (
            "Is fact A present?\nIs fact B present?",
            "Is fact A present?\nIs fact A present?\nIs fact B present?",
        ):
            fake = model_with(text, "Is fact A present?", "yes", "yes", "no", "yes")
            result = await calculator(model=fake).calculate_async(
                prediction="candidate", reference="reference"
            )
            outcomes.append(result.value)
            self.assertEqual(cast(Any, fake.generate).await_count, 6)
            assert result.metadata is not None
            self.assertEqual(len(result.metadata["raw_prediction_answers"]), 2)
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0]["fixture Accuracy"], 0.5)

    async def test_malformed_answers_fail_through_public_calculator(self):
        fake = model_with("Does it work?", "Does it work?", "garbled", "garbled")
        with self.assertRaises(ValueError):
            await calculator(model=fake).calculate_async(
                prediction="candidate", reference="reference"
            )

    async def test_mode_and_protocol_are_persisted_with_score(self):
        fake = model_with("Does it work?", "Does it work?", "unknown", "unknown")
        result = await calculator(model=fake).calculate_async(
            prediction="candidate",
            input="source",
            reference="reference",
            metadata={"tag": "value"},
        )
        assert result.metadata is not None
        self.assertEqual(
            result.metadata["evaluation_protocol"], "evalsense.qags-derived.v2"
        )
        self.assertEqual(result.metadata["answer_comparison_mode"], "ternary")
        self.assertEqual(result.metadata["invalid_answer_policy"], "fail_sample")
        self.assertEqual(
            result.metadata["question_weighting"], "uniform_unique_questions"
        )
        assert result.explanation is not None
        self.assertIn("QAGS", result.explanation)
        for call in cast(Any, fake.generate).await_args_list:
            prompt = json.loads(call.args[0])
            self.assertEqual(prompt["input"], "source")
            self.assertEqual(prompt["metadata"], {"tag": "value"})
            self.assertEqual(prompt["prediction"], "candidate")
            self.assertEqual(prompt["reference"], "reference")

    async def test_judge_mode_rejects_malformed_comparison_before_weighting(self):
        for text in (
            "maybe",
            "unknown",
            "yes or no",
            "The answer is yes",
            "",
            "no comparison possible",
        ):
            for logprobs in (True, False):
                with (
                    self.subTest(text=text, logprobs=logprobs),
                    self.assertRaises(ValueError),
                ):
                    await calculator(
                        "judge", model_with(text), logprobs
                    )._evaluate_judge_answers(
                        prediction="candidate",
                        questions=["One?"],
                        raw_prediction_answers=["alpha"],
                        raw_reference_answers=["beta"],
                        score_metadata={},
                    )

    async def test_judge_mode_valid_binary_comparisons(self):
        for text, expected in [("Yes.", 1.0), ("false", 0.0)]:
            with self.subTest(text=text):
                score = await calculator(
                    "judge", model_with(text)
                )._evaluate_judge_answers(
                    prediction="candidate",
                    questions=["One?"],
                    raw_prediction_answers=["alpha"],
                    raw_reference_answers=["beta"],
                    score_metadata={},
                )
                self.assertEqual(score.value, expected)

    async def test_judge_weighting_still_requires_valid_text(self):
        for text, probability in [("yes", 0.75), ("no", 0.25)]:
            with (
                self.subTest(text=text),
                patch(
                    "evalsense.evaluation.evaluators.qags.extract_weighted_binary_answer",
                    return_value=probability,
                ),
            ):
                result = await calculator(
                    "judge", model_with(text), True
                )._evaluate_judge_answers(
                    prediction="candidate",
                    questions=["One?"],
                    raw_prediction_answers=["alpha"],
                    raw_reference_answers=["beta"],
                    score_metadata={},
                )
                self.assertEqual(result.value, probability)

    async def test_missing_logprobs_preserves_valid_binary_fallback(self):
        with patch("evalsense.evaluation.evaluators.qags.logger.error"):
            result = await calculator(
                "judge", model_with("no"), True
            )._evaluate_judge_answers(
                prediction="candidate",
                questions=["One?"],
                raw_prediction_answers=["alpha"],
                raw_reference_answers=["beta"],
                score_metadata={},
            )
        self.assertEqual(result.value, 0.0)

    async def test_judge_rejects_empty_or_incomplete_batches_without_calls(self):
        for questions, pa, ra in [
            ([], [], []),
            (["One?"], [""], [""]),
            (["One?"], ["alpha"], []),
            (["One?"], ["alpha"], ["beta", "gamma"]),
        ]:
            fake = model_with()
            with (
                self.subTest(questions=questions, pa=pa, ra=ra),
                self.assertRaises(ValueError),
            ):
                await calculator("judge", fake)._evaluate_judge_answers(
                    prediction="candidate",
                    questions=questions,
                    raw_prediction_answers=pa,
                    raw_reference_answers=ra,
                    score_metadata={},
                )
            cast(Any, fake.generate).assert_not_awaited()

    async def test_exact_public_path_uses_filtered_question_set(self):
        fake = model_with("1. Where?\n2. Where?", "Where?", "London", " LONDON ")
        result = await calculator("exact", fake).calculate_async(
            prediction="candidate", reference="reference"
        )
        self.assertEqual(result.value, 1.0)
        self.assertEqual(cast(Any, fake.generate).await_count, 4)

    async def test_judge_public_path_uses_filtered_question_set(self):
        fake = model_with("Where?", "Where?", "London", "London, England", "Yes.")
        result = await calculator("judge", fake).calculate_async(
            prediction="candidate", reference="reference"
        )
        self.assertEqual(result.value, 1.0)
        self.assertEqual(cast(Any, fake.generate).await_count, 5)

    async def test_evaluator_names_match_scorer_and_component_keys(self):
        for mode in ("ternary", "exact", "judge"):
            for name, model_name in (
                ("Custom agreement", "Judge A"),
                ("Other criterion", "Judge B"),
            ):
                with self.subTest(mode=mode, name=name):
                    config = FixtureConfig(
                        cast(Literal["ternary", "exact", "judge"], mode), logprobs=False
                    )
                    evaluator = get_qags_evaluator(
                        name=name,
                        model_name=model_name,
                        config=config,
                        model_config=ModelConfig("mockllm/judge"),
                    )
                    factory = evaluator.scorer
                    assert isinstance(factory, QagsScorerFactory)
                    self.assertEqual(evaluator.name, f"{name} ({model_name})")
                    self.assertEqual(factory.name, evaluator.name)
                    responses = ["One?", "One?", "yes", "yes"]
                    if mode == "judge":
                        responses.append("yes")
                    score_fun = factory.create_scorer(model_with(*responses))
                    state = TaskState(
                        model=ModelName("mockllm/generator"),
                        sample_id="fixture",
                        epoch=1,
                        input="input",
                        messages=[],
                        output=ModelOutput.from_content(
                            model="fixture", content="candidate"
                        ),
                    )
                    result = await score_fun(state, Target("reference"))
                    assert result is not None
                    if mode == "ternary":
                        keys = {
                            f"{evaluator.name} {metric}"
                            for metric in ("Coverage", "Groundedness", "Accuracy")
                        }
                        self.assertEqual(result.value, dict.fromkeys(keys, 1.0))
                        assert isinstance(factory.metrics, list)
                        declared_metrics = factory.metrics[0]
                        assert isinstance(declared_metrics, dict)
                        self.assertEqual(set(declared_metrics), keys)
                    else:
                        self.assertEqual(result.value, 1.0)


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, directory: str, malformed: bool = False):
        manager = HuggingFaceDatasetManager(
            "qa-fixture", splits=["test"], version="v1", data_dir=directory
        )
        DatasetDict(
            {
                "test": Dataset.from_dict(
                    {"input": ["source"], "target": ["reference"], "id": ["sample"]}
                )
            }
        ).save_to_disk(manager.main_data_path)
        fake = model_with(
            "Does it work?",
            "Does it work?",
            "garbled" if malformed else "yes",
            "garbled" if malformed else "yes",
        )
        config = FixtureConfig("ternary", logprobs=False)
        declared = get_qags_evaluator(
            config=config, model_config=ModelConfig("mockllm/judge")
        )
        factory = cast(QagsScorerFactory, declared.scorer)
        evaluator = Evaluator(declared.name, factory.create_scorer(fake))
        project = Project("qags-fixture")
        experiment = ExperimentConfig(
            manager,
            GenerationSteps("generate", generate()),
            ModelConfig("mockllm/model"),
            evaluator=evaluator,
        )
        Pipeline(experiment, project).run(
            show_progress=False, eval_kwargs={"display": "none", "log_format": "json"}
        )
        return project, experiment

    def test_valid_scores_survive_pipeline_and_log_roundtrip(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "evalsense.workflow.project.PROJECTS_PATH", Path(directory) / "projects"
            ),
        ):
            project, experiment = self.run_pipeline(directory)
            record = project.get_record(experiment.evaluation_record)
            assert record is not None
            self.assertEqual(record.status, "success")
            result = project.get_log(experiment.evaluation_record)
            assert result is not None and result.samples and result.samples[0].scores
            score = next(iter(result.samples[0].scores.values()))
            assert isinstance(score.value, dict)
            self.assertTrue(all(value == 1.0 for value in score.value.values()))
            self.assertTrue(
                experiment.evaluation_record.evaluator_name.startswith("QAGS (")
            )
            for suffix in (".json", ".eval"):
                path = Path(directory) / f"roundtrip{suffix}"
                write_eval_log(result, location=str(path))
                restored = read_eval_log(str(path))
                assert restored.samples and restored.samples[0].scores
                restored_score = next(iter(restored.samples[0].scores.values()))
                self.assertEqual(restored_score.value, score.value)
                assert restored_score.metadata is not None
                self.assertEqual(
                    restored_score.metadata["evaluation_protocol"],
                    "evalsense.qags-derived.v2",
                )
                self.assertEqual(
                    restored_score.metadata["question_filtering"]["retained_questions"],
                    1,
                )

    def test_malformed_answers_do_not_publish_a_successful_score(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "evalsense.workflow.project.PROJECTS_PATH", Path(directory) / "projects"
            ),
        ):
            project, experiment = self.run_pipeline(directory, malformed=True)
            record = project.get_record(experiment.evaluation_record)
            assert record is not None
            self.assertEqual(record.status, "error")
            result = project.get_log(experiment.evaluation_record)
            assert result is not None and result.samples
            self.assertFalse(result.samples[0].scores)


if __name__ == "__main__":
    unittest.main()
