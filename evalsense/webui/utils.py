import ast
import json
from typing import Any, Callable, TypedDict

import gradio as gr

from evalsense.constants import PROJECTS_PATH
from evalsense.utils.credentials import ModelCredentialError, validate_public_config
from evalsense.webui.state import AppState

type GradioInput = gr.Checkbox | gr.Dropdown | gr.Radio | gr.Textbox


class ListenerConfig(TypedDict):
    """Configuration for a textbox listener.

    Attributes:
        state_field (str): The name of the state field to update.
        parser (Callable[[str], Any] | None): An optional parser function
            to process the input value.
    """

    state_field: str
    parser: Callable[[str], Any] | None


def empty_is_none_parser_for(type: type) -> Callable[[str], Any | None]:
    """Returns a parser function that returns None for empty strings.

    Args:
        type (type): The type of the value to parse.

    Returns:
        Callable[[str], Any | None]: The parser function.
    """

    def parser(input_string: str) -> Any | None:
        if not input_string:
            return None
        try:
            return type(input_string)
        except Exception:
            raise ValueError(f"Unable to parse {input_string} as {type.__name__}.")

    return parser


def list_parser(input_string: str) -> list[str]:
    """Parses a comma-separated string into a list of strings.

    Arguments:
        input_string (str): The input string to parse.

    Returns:
        list[str]: A list containing the parsed strings.
    """
    return input_string.replace(" ", "").split(",")


def dict_parser(input_string: str) -> dict[str, Any]:
    """Parse public model options without echoing rejected input.

    Credential fields and unsupported values are rejected before session storage.

    Arguments:
        input_string (str): The input string to parse.

    Returns:
        dict[str, Any]: The parsed dictionary.
    """
    if not input_string:
        return {}
    try:
        value = ast.literal_eval(input_string)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        raise gr.Error(
            "Invalid dictionary format. Enter a Python dictionary without credentials."
        ) from None
    if type(value) is not dict:
        raise gr.Error("Expected a Python dictionary without credentials.")
    validate_public_config(value)
    return value


def configuration_for_display(value: dict[str, Any]) -> dict[str, Any]:
    """Hide unsafe legacy UI state rather than echoing it into JSON components."""
    try:
        if any(
            key in value and type(value[key]) is not dict
            for key in ("model_args", "generation_args")
        ):
            raise ModelCredentialError(
                "Model argument containers must be dictionaries."
            )
        validate_public_config(value)
    except ModelCredentialError:
        return {
            "configuration": "Hidden: remove this configuration and use provider-managed authentication."
        }
    return json.loads(json.dumps(value))


def setup_listeners(
    listener_config: dict[GradioInput, ListenerConfig],
    state: gr.State,
):
    """Sets up listeners updating the application state based on user inputs.

    Arguments:
        listener_config (dict[GradioInput, TextboxListenerConfig]): The configuration
            specifying the parsers for processing user inputs and the corresponding
            state fields to update.
        state (gr.State): The current state of the Gradio application.
    """
    for input_element, element_config in listener_config.items():

        @input_element.change(inputs=[input_element, state], outputs=[state])
        def update_field(
            entered_value: str,
            state: AppState,
            config: ListenerConfig = element_config,
        ):
            value = entered_value
            if config["parser"] is not None:
                value = config["parser"](entered_value)
            state[config["state_field"]] = value
            return state


def discover_projects(state: AppState) -> AppState:
    """Discovers existing evaluation projects in the projects directory.

    Returns:
        AppState: The updated application state with the list of existing projects.
    """
    try:
        projects = [entry.name for entry in PROJECTS_PATH.iterdir() if entry.is_dir()]
    except FileNotFoundError:
        projects = []
    state["existing_projects"] = projects
    return state
