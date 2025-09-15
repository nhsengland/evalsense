import gradio as gr
from pandas import DataFrame
import traceback

from evalsense.datasets import DatasetManager
from evalsense.webui.utils import (
    GradioInput,
    ListenerConfig,
    empty_is_none_parser_for,
    setup_listeners,
    list_parser,
)
from evalsense.webui.state import AppState


def data_tab(state: gr.State):
    """Renders the data tab user interface.

    Arguments:
        state (gr.State): The current state of the Gradio application.
    """

    # Data tab user interface
    gr.Markdown("Use this tab to configure the dataset for evaluation.")
    gr.Markdown(
        "**Tip:** You don't need to load a dataset before using it for evaluation, but doing so allows you to preview it below."
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Dataset Configuration")
            gr.Markdown(
                "You can use a dedicated EvalSense dataset or load one from Hugging Face Hub."
            )
            dataset_name_input = gr.Textbox(
                label="Dataset Name", info="The name of the dataset to use."
            )
            data_splits_input = gr.Textbox(
                label="Data Splits (comma-separated)",
                info="The dataset splits to use, separated by commas.",
            )
            dataset_version_input = gr.Textbox(
                label="Dataset Version",
                info="Leave this blank to use the default version",
            )
            sample_limit_input = gr.Number(
                value=5,
                precision=0,
                label="Sample Limit",
                info="The number of samples to display below when loading the dataset.",
            )
            dataset_load_button = gr.Button("Load Dataset", variant="primary")
        with gr.Column():
            gr.Markdown("## Field Configuration")
            gr.Markdown(
                "If you are not using some of these fields in your evaluation, you can safely leave them set to their default values."
            )
            input_field_name_input = gr.Textbox(
                label="Input Field Name",
                value=state.value["input_field_name"],
                info="The name of the field containing the main input",
            )
            target_field_name_input = gr.Textbox(
                label="Target Field Name",
                value=state.value["target_field_name"],
                info="The name of the field containing the target output",
            )
            choices_field_name_input = gr.Textbox(
                label="Choices Field Name",
                value=state.value["choices_field_name"],
                info="The name of the field containing the list of answer choices (for multiple-choice questions)",
            )
            id_field_name_input = gr.Textbox(
                label="ID Field Name",
                value=state.value["id_field_name"],
                info="The name of the field containing the unique ID for each sample",
            )
            metadata_fields_input = gr.Textbox(
                label="Metadata Fields (comma-separated)",
                info="List of additional field names that should be used as metadata",
            )
    with gr.Row():
        with gr.Column():
            total_samples_indicator = gr.Markdown("Total samples: **?**")
            sample_df = gr.Dataframe(
                label="Sample Data",
                headers=["Load a dataset to see sample data"],
                col_count=1,
                interactive=False,
            )

    # Textbox listeners
    LISTENER_CONFIG: dict[GradioInput, ListenerConfig] = {
        dataset_name_input: {
            "state_field": "dataset_name",
            "parser": None,
        },
        data_splits_input: {
            "state_field": "dataset_splits",
            "parser": list_parser,
        },
        dataset_version_input: {
            "state_field": "dataset_version",
            "parser": empty_is_none_parser_for(str),
        },
        input_field_name_input: {
            "state_field": "input_field_name",
            "parser": None,
        },
        target_field_name_input: {
            "state_field": "target_field_name",
            "parser": None,
        },
        choices_field_name_input: {
            "state_field": "choices_field_name",
            "parser": None,
        },
        id_field_name_input: {"state_field": "id_field_name", "parser": None},
        metadata_fields_input: {
            "state_field": "metadata_fields",
            "parser": list_parser,
        },
    }
    setup_listeners(LISTENER_CONFIG, state)

    @dataset_load_button.click(
        inputs=[state, sample_limit_input],
        outputs=[state, total_samples_indicator, sample_df],
    )
    def load_dataset(
        state: AppState,
        sample_limit: int,
        progress=gr.Progress(),
    ):
        # Load dataset
        gr.Info("Loading dataset — this may take a while...")
        try:
            dataset_manager = DatasetManager.create(
                state["dataset_name"],
                splits=list(state["dataset_splits"]),
                version=state["dataset_version"],
            )
            dataset = dataset_manager.load()
            sample_df = dataset.select(
                range(min(len(dataset), sample_limit))
            ).to_pandas()
            assert isinstance(sample_df, DataFrame), "Expected a DataFrame"
        except Exception as e:
            traceback.print_exc()
            raise gr.Error(f"Failed to load dataset: {e}")

        # Prepare new dataframe display
        total_samples = f"Total samples: **{len(dataset)}**"
        df_display = gr.DataFrame(
            value=sample_df,
            label="Sample data",
            visible=True,
        )
        gr.Success("Dataset successfully loaded!")

        return state, total_samples, df_display
