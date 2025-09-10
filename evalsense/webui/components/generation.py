import gradio as gr

from evalsense.webui.components.utils import (
    TextboxListenerConfig,
    setup_textbox_listeners,
)


def generation_tab(state: gr.State):
    # Generation tab user interface
    gr.Markdown("Use this tab to configure the prompt to use during the generation.")
    gr.Markdown("### Prompt Configuration")
    system_prompt_input = gr.TextArea(
        label="System Prompt",
        info="The prompt to use for the system message. You can use Python f-string format to substitute the main input into a `{prompt}` placeholder, as well as for definiting placeholders for any additional metadata fields specified on the data tab.",
        max_lines=15,
    )
    user_prompt_input = gr.TextArea(
        label="User Prompt",
        info="The prompt to use for the user message. You can use Python f-string format to substitute the main input into a `{prompt}` placeholder, as well as for definiting placeholders for any additional metadata fields specified on the data tab.",
        max_lines=15,
    )

    LISTENER_CONFIG: dict[gr.Textbox, TextboxListenerConfig] = {
        system_prompt_input: {"state_field": "system_prompt", "parser": None},
        user_prompt_input: {"state_field": "user_prompt", "parser": None},
    }
    setup_textbox_listeners(LISTENER_CONFIG, state)
