from dataclasses import replace
from typing import Any, Callable, TypedDict

import gradio as gr

from evalsense.webui.state import AppState


class TextboxListenerConfig(TypedDict):
    state_field: str
    parser: Callable[[str], Any] | None


def tuple_parser(input_string: str) -> tuple[str, ...]:
    return tuple(input_string.replace(" ", "").split(","))


def setup_textbox_listeners(
    listener_config: dict[gr.Textbox, TextboxListenerConfig],
    state: gr.State,
):
    for input_element, element_config in listener_config.items():

        def create_listener(
            input_element: gr.Textbox,
            element_config: TextboxListenerConfig,
        ):
            @input_element.change(inputs=[input_element, state], outputs=[state])
            def update_field(entered_value: str, state: AppState):
                value = entered_value
                if element_config["parser"] is not None:
                    value = element_config["parser"](entered_value)
                return replace(state, **{element_config["state_field"]: value})

            return update_field

        create_listener(input_element, element_config)
