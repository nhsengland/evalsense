from typing import Any, TypedDict


class AppModelConfig(TypedDict):
    model_name: str
    model_args: dict[str, Any]
    generation_args: dict[str, Any]


class AppState(TypedDict):
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


def get_initial_state() -> AppState:
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
    }
