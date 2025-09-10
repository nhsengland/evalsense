import secrets

import gradio as gr
from gradio.themes import Default

from evalsense.webui.components.data import data_tab
from evalsense.webui.components.execution import execution_tab
from evalsense.webui.components.generation import generation_tab
from evalsense.webui.state import AppState

theme = Default(primary_hue="blue")

with gr.Blocks(theme=theme) as demo:
    state = gr.State(AppState())
    gr.Markdown("# 🔎 EvalSense")
    gr.Markdown(
        "To run an evaluation, configure the settings on the individual tabs and start it from the **Execution** tab. For EvalSense documentation and guidance regarding the available evaluation metrics, please visit the [EvalSense homepage](https://nhsengland.github.io/evalsense/)."
    )
    with gr.Tab("Data"):
        data_tab(state)
    with gr.Tab("Generation"):
        generation_tab(state)
    with gr.Tab("Models"):
        pass
    with gr.Tab("Evaluators"):
        pass
    with gr.Tab("Execution"):
        execution_tab(state)

if __name__ == "__main__":
    print("* Server username: user")
    password = secrets.token_urlsafe(20)
    print(f"* Server password: {password}")
    demo.launch(share=False, auth=("user", password))
