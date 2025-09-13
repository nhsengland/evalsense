import gradio as gr
import pandas as pd

from evalsense.webui.state import AppState
from evalsense.workflow import Project
from evalsense.workflow.analysers import (
    CorrelationResults,
    MetricCorrelationAnalyser,
    TabularResultAnalyser,
)


def results_tab(state: gr.State):
    """Renders the results tab user interface."""
    gr.Markdown("Use this tab to preview the results of the evaluation.")
    gr.Markdown("## Evaluation Results")

    @gr.render(inputs=[state])
    def show_project_dropdown(local_state: AppState):
        project_name_input = gr.Dropdown(
            label="Project Name",
            info="The name of the evaluation project for which the results should be displayed.",
            choices=local_state["existing_projects"],
            value=None,
        )
        load_button = gr.Button("Load Project", variant="primary")
        results_df = gr.DataFrame(
            label="Evaluation Results",
            headers=["Load a project to see results"],
            col_count=1,
            interactive=False,
        )
        metric_correlation = gr.Plot(label="Metric Correlation", format="png")

        @load_button.click(
            inputs=[project_name_input], outputs=[results_df, metric_correlation]
        )
        def load_project(project_name: str):
            project = Project(project_name)

            tabular_analyser = TabularResultAnalyser[pd.DataFrame](
                output_format="pandas"
            )
            summary_results = tabular_analyser(project)
            summary_results.sort_values(
                by="model", inplace=True, key=lambda col: col.str.lower()
            )
            summary_results = summary_results.round(2)

            correlation_analyser = MetricCorrelationAnalyser[
                CorrelationResults[pd.DataFrame]
            ](output_format="pandas")
            correlation_results = correlation_analyser(
                project, return_plot=True, figsize=(9, 7)
            )
            plot = correlation_results.figure
            assert plot, "Correlation plot cannot be None"

            return summary_results, plot
