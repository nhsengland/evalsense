import ast
from typing import Any, Callable, TypedDict

import gradio as gr

from evalsense.webui.state import AppState


class TextboxListenerConfig(TypedDict):
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


def tuple_parser(input_string: str) -> tuple[str, ...]:
    """Parses a comma-separated string into a tuple of strings.

    Arguments:
        input_string (str): The input string to parse.

    Returns:
        tuple[str, ...]: A tuple containing the parsed strings.
    """
    return tuple(input_string.replace(" ", "").split(","))


def dict_parser(input_string: str) -> dict[str, Any]:
    """Parses a string representation of a dictionary into an actual dictionary.

    Arguments:
        input_string (str): The input string to parse.

    Returns:
        dict[str, Any]: The parsed dictionary.
    """
    if not input_string:
        return {}
    try:
        return ast.literal_eval(input_string)
    except Exception:
        raise gr.Error(f"Invalid dictionary format: {input_string}")


def setup_textbox_listeners(
    listener_config: dict[gr.Textbox, TextboxListenerConfig],
    state: gr.State,
):
    """Sets up listeners updating the application state for the supplied textboxes.

    Arguments:
        listener_config (dict[gr.Textbox, TextboxListenerConfig]): The configuration
            for the textboxes.
        state (gr.State): The current state of the Gradio application.
    """
    for input_element, element_config in listener_config.items():

        @input_element.change(inputs=[input_element, state], outputs=[state])
        def update_field(
            entered_value: str,
            state: AppState,
            config: TextboxListenerConfig = element_config,
        ):
            value = entered_value
            if config["parser"] is not None:
                value = config["parser"](entered_value)
            state[config["state_field"]] = value
            return state
