import gradio as gr
import pandas as pd

from evalsense.webui.state import AppState
from evalsense.workflow import Project
from evalsense.workflow.analysers import (
    CorrelationResults,
    MetricCorrelationAnalyser,
    TabularResultAnalyser,
)
from evalsense.workflow.analysers.meta_result_analyser import MetaResultAnalyser


def load_project(project_name: str, is_meta_eval: bool, expected_meta_tiers: str = ""):
    """Loads the project and returns the summary results and correlation plot."""
    try:
        project = Project(project_name)

        if is_meta_eval:
            if not expected_meta_tiers.strip():
                raise ValueError(
                    "Enter the complete tier IDs from the experiment design."
                )
            try:
                tiers = [int(value.strip()) for value in expected_meta_tiers.split(",")]
            except ValueError:
                raise ValueError(
                    "Expected tiers must be comma-separated integers."
                ) from None
            tabular_analyser = MetaResultAnalyser[pd.DataFrame](output_format="pandas")
            summary_results = tabular_analyser(
                project,
                meta_tier_field="perturbation_tier",
                lower_tier_is_better=True,
                expected_tiers=tiers,
            )
            summary_results.sort_values(
                by=["avg_correlation"], inplace=True, ascending=False
            )
            plot = None
        else:
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
    except Exception as e:
        raise gr.Error(f"Error loading results: {type(e).__name__}: {e}")

    return summary_results, plot if plot is not None else gr.update(visible=False)


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
        is_meta_eval_input = gr.Radio(
            label="Evaluation Type",
            info="What type of evaluation was performed in this project?",
            value=False,
            choices=[
                ("Standard Evaluation", False),
                ("Meta-Evaluation", True),
            ],
        )
        expected_meta_tiers_input = gr.Textbox(
            label="Expected meta-evaluation tiers",
            info="Enter all planned tier IDs, for example 0, 1, 2. Do not infer the set from successful results. At least two tiers are required; two-tier designs report pairwise ordering.",
            value="",
            visible=False,
        )
        meta_notice = gr.Markdown(
            "These correlations describe agreement with intended perturbation tiers. "
            "They do not verify realised severity or establish evaluator validity. "
            "Review exclusions and confidence-interval status before comparing results. "
            "For two tiers, pairwise_accuracy excludes ties (n_constant); interval "
            "limits are on the avg_correlation scale. "
            "Counts cover IDs present in available logs; use the Python API with a "
            "sample manifest to count samples missing from every log.",
            visible=False,
        )
        is_meta_eval_input.change(
            fn=lambda is_meta: (gr.update(visible=is_meta), gr.update(visible=is_meta)),
            inputs=[is_meta_eval_input],
            outputs=[expected_meta_tiers_input, meta_notice],
        )
        load_button = gr.Button("Load Project", variant="primary")
        results_df = gr.DataFrame(
            label="Evaluation Results",
            headers=["Load a project to see results"],
            column_count=1,
            interactive=False,
        )
        metric_correlation = gr.Plot(label="Metric Correlation", format="png")

        load_button.click(
            inputs=[is_meta_eval_input],
            outputs=[metric_correlation],
            fn=lambda is_meta: gr.update(visible=not is_meta),
        ).then(
            fn=load_project,
            inputs=[project_name_input, is_meta_eval_input, expected_meta_tiers_input],
            outputs=[results_df, metric_correlation],
        )
