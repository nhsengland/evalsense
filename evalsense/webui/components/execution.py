import traceback
import gradio as gr

from evalsense.webui.execution import execute_evaluation
from evalsense.webui.state import AppState
from evalsense.webui.utils import setup_listeners


def run_evaluation(state: AppState):
    try:
        execute_evaluation(state)
    except Exception as e:
        gr.Warning(f"Error during evaluation: {type(e).__name__}: {e}")
        traceback.print_exc()
        return f"❌ Evaluation failed with an error:\n{type(e).__name__}: {e}"
    return "✅ Evaluation completed successfully."


def execution_tab(state: gr.State):
    """Renders the execution tab user interface.

    Arguments:
        state (gr.State): The current state of the Gradio application.
    """
    gr.Markdown(
        "Use this tab for configuring the evaluation project and the final execution."
    )
    gr.Markdown("## Project")
    project_name_input = gr.Textbox(
        label="Project Name",
        value=state.value["project_name"],
        info="The name of the evaluation project.",
    )
    setup_listeners(
        {project_name_input: {"state_field": "project_name", "parser": None}}, state
    )

    gr.Markdown("## Current Configuration")
    gr.JSON(
        lambda state: state, inputs=[state], label="Current Configuration", open=True
    )

    gr.Markdown("## Run Evaluation")
    gr.Markdown(
        "Once you are satisfied with your configuration, you can start the evaluation by clicking the button below. When the evaluation is running, you can monitor its progress and view the detailed logs in the terminal from which you started the server. After the evaluation is complete, you can view the results in the **Results** tab."
    )
    run_button = gr.Button("Run Evaluation", variant="primary")

    evaluation_status = gr.Textbox(
        label="Evaluation Status",
        value="ℹ️ Evaluation not yet started.",
        interactive=False,
    )

    run_button.click(fn=lambda: gr.update(interactive=False), outputs=run_button).then(
        fn=run_evaluation, inputs=[state], outputs=[evaluation_status]
    ).then(fn=lambda: gr.update(interactive=True), outputs=run_button)
