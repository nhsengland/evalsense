from dataclasses import asdict

import gradio as gr


def execution_tab(state: gr.State):
    gr.Markdown("Use this tab to execute the selected evaluation.")
    gr.JSON(lambda state: asdict(state), inputs=[state], label="Current State")
