from typing import Any, TypedDict


class AppModelConfig(TypedDict):
    """Model configuration to be used within the Gradio application.

    Attributes:
        model_name (str): The name of the model to use.
        model_args (dict[str, Any]): The arguments to pass to the model.
        generation_args (dict[str, Any]): The arguments to use for text generation.
    """

    model_name: str
    model_args: dict[str, Any]
    generation_args: dict[str, Any]


class AppEvaluatorConfig(TypedDict):
    """Evaluator configuration to be used within the Gradio application.

    Attributes:
        evaluator_name (str): The name of the evaluator to use.
        evaluator_args (dict[str, Any]): The arguments to pass to the evaluator.
    """

    evaluator_name: str
    evaluator_args: dict[str, Any]


class AppState(TypedDict):
    """Application state to be used within the Gradio application.

    Attributes:
        dataset_name (str): The name of the dataset to evaluate on.
        dataset_splits (tuple[str]): The used splits of the dataset.
        dataset_version (str): The used version of the dataset.
        input_field_name (str): The name of the main input field in the dataset.
        target_field_name (str): The name of the target field in the dataset.
        choices_field_name (str): The name of the answer choices field in the dataset.
        id_field_name (str): The name of the ID field in the dataset.
        metadata_fields (tuple[str]): The names of the metadata fields in the dataset.
        system_prompt (str): The system prompt to use for generation.
        user_prompt (str): The user prompt to use for generation.
        model_configs (list[AppModelConfig]): The model configurations to use for
            generation.
        evaluator_configs (list[AppEvaluatorConfig]): The evaluator configurations
            to use for evaluation.
    """

    dataset_name: str
    dataset_splits: tuple[str]
    dataset_version: str
    input_field_name: str
    target_field_name: str
    choices_field_name: str
    id_field_name: str
    metadata_fields: tuple[str]
    system_prompt: str
    user_prompt: str
    model_configs: list[AppModelConfig]
    evaluator_configs: list[AppEvaluatorConfig]


def get_initial_state() -> AppState:
    """Provides the initial application state.

    Returns:
        AppState: The initial application state.
    """
    return {
        "dataset_name": "",
        "dataset_splits": tuple(),
        "dataset_version": "",
        "input_field_name": "input",
        "target_field_name": "target",
        "choices_field_name": "choices",
        "id_field_name": "id",
        "metadata_fields": tuple(),
        "system_prompt": "",
        "user_prompt": "",
        "model_configs": list(),
        "evaluator_configs": list(),
    }
