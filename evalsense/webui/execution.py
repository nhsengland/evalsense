from typing import Any

from inspect_ai.dataset import FieldSpec, Sample
from inspect_ai.model import GenerateConfigArgs
from inspect_ai.solver import generate, prompt_template, system_message

from evalsense.datasets import DatasetManager
from evalsense.evaluation import (
    Evaluator,
    ExperimentBatchConfig,
    TaskConfig,
)
from evalsense.generation import ModelConfig, GenerationSteps
from evalsense.tasks.task_preprocessor import DefaultTaskPreprocessor
from evalsense.webui.configurators import EvaluatorConfigurator
from evalsense.webui.state import AppState
from evalsense.workflow import Pipeline, Project


def get_dataset_manager(state: AppState) -> DatasetManager:
    """Creates and returns a DatasetManager based on the current application state.

    Args:
        state (AppState): The current application state.

    Returns:
        DatasetManager: The instantiated DatasetManager.
    """
    return DatasetManager.create(
        name=state["dataset_name"],
        splits=state["dataset_splits"],
        version=state["dataset_version"],
    )


def get_model_configs(state: AppState) -> list[ModelConfig]:
    """Creates and returns a list of ModelConfig based on the current application state.

    Args:
        state (AppState): The current application state.

    Returns:
        list[ModelConfig]: The list of instantiated ModelConfig objects.
    """
    return [
        ModelConfig(
            m["model_name"],
            model_args=m["model_args"],
            generation_args=GenerateConfigArgs(**m["generation_args"]),
        )
        for m in state["model_configs"]
    ]


def get_evaluators(state: AppState) -> list[Evaluator]:
    """Creates and returns a list of Evaluator instances based on the current application state.

    Args:
        state (AppState): The current application state.

    Returns:
        list[Evaluator]: The list of instantiated Evaluator objects.
    """
    evaluators: list[Evaluator] = []
    for evaluator_config in state["evaluator_configs"]:
        configurator = EvaluatorConfigurator.create(evaluator_config["evaluator_name"])
        evaluator = configurator.instantiate_evaluator(
            **evaluator_config["evaluator_args"]
        )
        evaluators.append(evaluator)
    return evaluators


def execute_standard_evaluation(state: AppState):
    """Executes a standard evaluation for the given application state.

    Args:
        state (AppState): The current application state.
    """
    dataset_manager = get_dataset_manager(state)
    generation_steps = GenerationSteps(
        name=state["generation_steps_name"],
        steps=[
            system_message(state["system_prompt"]),
            prompt_template(state["user_prompt"]),
            generate(),
        ],
    )
    field_spec = FieldSpec(
        input=state["input_field_name"],
        target=state["target_field_name"],
        choices=state["choices_field_name"],
        id=state["id_field_name"],
        metadata=state["metadata_fields"],
    )
    model_configs = get_model_configs(state)
    evaluators = get_evaluators(state)
    task_config = TaskConfig(
        dataset_manager=dataset_manager,
        generation_steps=generation_steps,
        field_spec=field_spec,
    )
    experiment_config = ExperimentBatchConfig(
        tasks=[task_config], model_configs=model_configs, evaluators=evaluators
    )
    project = Project(name=state["project_name"])
    pipeline = Pipeline(experiments=experiment_config, project=project)
    pipeline.run()


def execute_meta_evaluation(state: AppState):
    """Executes a meta-evaluation based on the current application state.

    Args:
        state (AppState): The current application state.
    """
    dataset_manager = get_dataset_manager(state)

    tasks = []
    for tier_id, perturbation_tier_subprompt in enumerate(
        state["perturbation_tier_subprompts"]
    ):
        system_prompt = state["system_prompt"].replace(
            "{perturbation_tier_subprompt}", perturbation_tier_subprompt
        )
        user_prompt = state["user_prompt"].replace(
            "{perturbation_tier_subprompt}", perturbation_tier_subprompt
        )
        generation_steps = GenerationSteps(
            name=f"{state['generation_steps_name']} (Tier {tier_id + 1})",
            steps=[
                system_message(system_prompt),
                prompt_template(user_prompt),
                generate(),
            ],
        )

        # We use a RecordToSample function to add the perturbation tier
        # to the metadata
        def perturbation_record_to_sample(
            record: dict[str, Any],
            tier_id: int = tier_id,
        ) -> Sample:
            return Sample(
                input=record[state["input_field_name"]],
                target=record.get(state["target_field_name"], ""),
                choices=record.get(state["choices_field_name"]),
                id=record.get(state["id_field_name"]),
                metadata={k: record[k] for k in state["metadata_fields"]}
                | {"perturbation_tier": tier_id},
            )

        perturb_task_preprocessor = DefaultTaskPreprocessor(name="Perturbation")
        task_config = TaskConfig(
            dataset_manager=dataset_manager,
            generation_steps=generation_steps,
            field_spec=perturbation_record_to_sample,
            task_preprocessor=perturb_task_preprocessor,
        )
        tasks.append(task_config)

    model_configs = get_model_configs(state)
    evaluators = get_evaluators(state)
    experiment_config = ExperimentBatchConfig(
        tasks=tasks, model_configs=model_configs, evaluators=evaluators
    )
    project = Project(name=state["project_name"])
    pipeline = Pipeline(experiments=experiment_config, project=project)
    pipeline.run()


def execute_evaluation(state: AppState):
    """Executes the evaluation based on the current application state.

    Args:
        state (AppState): The current application state.
    """
    if state["is_meta_eval"]:
        execute_meta_evaluation(state)
    else:
        execute_standard_evaluation(state)
