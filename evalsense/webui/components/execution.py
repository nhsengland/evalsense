import gradio as gr


def execution_tab(state: gr.State):
    """Renders the execution tab user interface.

    Arguments:
        state (gr.State): The current state of the Gradio application.
    """
    gr.Markdown("Use this tab to execute the selected evaluation.")
    gr.JSON(lambda state: state, inputs=[state], label="Current State", open=True)
