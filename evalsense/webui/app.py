import secrets

import gradio as gr
from gradio.themes import Default

from evalsense.webui.components.data import data_tab
from evalsense.webui.components.evaluators import evaluators_tab
from evalsense.webui.components.execution import execution_tab
from evalsense.webui.components.generation import generation_tab
from evalsense.webui.components.models import models_tab
from evalsense.webui.components.results import results_tab
from evalsense.webui.state import get_initial_state
from evalsense.webui.utils import discover_projects


def launch_webui(
    password: str | None = None, no_auth: bool = False, share: bool = False
):
    """Launches the EvalSense Gradio web UI.

    Args:
        password: Password for authentication. If None, a random password is generated.
        no_auth: If True, disables authentication.
        share: If True, enables Gradio public sharing. This will make the app publicly
            accessible over the internet. Use with caution.
    """
    theme = Default(primary_hue="blue")
    with gr.Blocks(theme=theme, title="EvalSense") as demo:
        state = gr.State(get_initial_state())
        gr.Markdown("# 🔎 EvalSense")
        gr.Markdown(
            "To run an evaluation, configure its settings on the individual tabs and start it from the **Execution** tab. For EvalSense documentation and guidance regarding the available evaluation metrics, please visit the [EvalSense homepage](https://nhsengland.github.io/evalsense/)."
        )
        with gr.Tab("Data"):
            data_tab(state)
        with gr.Tab("Generation"):
            generation_tab(state)
        with gr.Tab("Models"):
            models_tab(state)
        with gr.Tab("Evaluators"):
            evaluators_tab(state)
        with gr.Tab("Execution"):
            execution_tab(state)
        with gr.Tab("Results"):
            results_tab(state)

        # Regularly discover projects and update the state
        timer = gr.Timer(3, active=True)
        timer.tick(fn=discover_projects, inputs=[state], outputs=[state])

    if share:
        print("* CAUTION: Gradio public sharing is enabled!")

    if no_auth:
        print("* CAUTION: Authentication disabled!")
        demo.launch(share=share)
    else:
        if password is None:
            password = secrets.token_urlsafe(20)
        print("* Server username: user")
        print(f"* Server password: {password}")
        demo.launch(share=share, auth=("user", password) if not no_auth else None)


if __name__ == "__main__":
    launch_webui()
