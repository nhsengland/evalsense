from inspect_ai.dataset import FieldSpec
from inspect_ai.model import GenerateConfigArgs
from inspect_ai.solver import generate, prompt_template, system_message

from evalsense.datasets import DatasetManager
from evalsense.evaluation import Evaluator, ExperimentBatchConfig, TaskConfig
from evalsense.generation import ModelConfig, GenerationSteps
from evalsense.webui.configurators import EvaluatorConfigurator
from evalsense.webui.state import AppState
from evalsense.workflow import Pipeline, Project


def execute_evaluation(state: AppState):
    """Executes the evaluation based on the current application state.

    Args:
        state (AppState): The current application state.
    """
    dataset_manager = DatasetManager.create(
        name=state["dataset_name"],
        splits=state["dataset_splits"],
        version=state["dataset_version"],
    )
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
    model_configs = [
        ModelConfig(
            m["model_name"],
            model_args=m["model_args"],
            generation_args=GenerateConfigArgs(**m["generation_args"]),
        )
        for m in state["model_configs"]
    ]
    evaluators: list[Evaluator] = []
    for evaluator_config in state["evaluator_configs"]:
        configurator = EvaluatorConfigurator.create(evaluator_config["evaluator_name"])
        evaluator = configurator.instantiate_evaluator(
            **evaluator_config["evaluator_args"]
        )
        evaluators.append(evaluator)

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
