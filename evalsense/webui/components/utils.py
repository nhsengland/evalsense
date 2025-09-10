import ast
from typing import Any, Callable, TypedDict

import gradio as gr

from evalsense.webui.state import AppState


class TextboxListenerConfig(TypedDict):
    state_field: str
    parser: Callable[[str], Any] | None


def tuple_parser(input_string: str) -> tuple[str, ...]:
    return tuple(input_string.replace(" ", "").split(","))


def dict_parser(input_string: str) -> dict[str, Any]:
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
