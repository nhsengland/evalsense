import gradio as gr


def execution_tab(state: gr.State):
    gr.Markdown("Use this tab to execute the selected evaluation.")
    gr.JSON(lambda state: state, inputs=[state], label="Current State", open=True)
