from dataclasses import replace
import gradio as gr
from pandas import DataFrame
import traceback
from typing import TypedDict

from evalsense.datasets import DatasetManager
from evalsense.webui.state import AppState


class TextFieldListenerConfig(TypedDict):
    state_field: str
    parse_as_tuple: bool


def data_tab(state: gr.State):
    # Data tab user interface
    gr.Markdown("Use this tab to configure the dataset for evaluation.")
    gr.Markdown(
        "**Tip:** You don't need to load a dataset before using it for evaluation, but doing so allows you to preview it below."
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Dataset Configuration")
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
            dataset_load_button = gr.Button("Load Dataset")
        with gr.Column():
            gr.Markdown("### Field Configuration")
            gr.Markdown(
                "If you are not using some of these fields in your evaluation, you can safely leave them set to their default values."
            )
            input_field_name_input = gr.Textbox(
                label="Input Field Name",
                value=state.value.input_field_name,
                info="The name of the field containing the main input",
            )
            target_field_name_input = gr.Textbox(
                label="Target Field Name",
                value=state.value.target_field_name,
                info="The name of the field containing the target output",
            )
            choices_field_name_input = gr.Textbox(
                label="Choices Field Name",
                value=state.value.choices_field_name,
                info="The name of the field containing the list of answer choices (for multiple-choice questions)",
            )
            id_field_name_input = gr.Textbox(
                label="ID Field Name",
                value=state.value.id_field_name,
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
            )

    # Configure text field listeners
    LISTENER_CONFIG: dict[gr.Textbox, TextFieldListenerConfig] = {
        dataset_name_input: {
            "state_field": "dataset_name",
            "parse_as_tuple": False,
        },
        data_splits_input: {
            "state_field": "dataset_splits",
            "parse_as_tuple": True,
        },
        dataset_version_input: {
            "state_field": "dataset_version",
            "parse_as_tuple": False,
        },
        input_field_name_input: {
            "state_field": "input_field_name",
            "parse_as_tuple": False,
        },
        target_field_name_input: {
            "state_field": "target_field_name",
            "parse_as_tuple": False,
        },
        choices_field_name_input: {
            "state_field": "choices_field_name",
            "parse_as_tuple": False,
        },
        id_field_name_input: {"state_field": "id_field_name", "parse_as_tuple": False},
        metadata_fields_input: {
            "state_field": "metadata_fields",
            "parse_as_tuple": True,
        },
    }

    for input_element, config in LISTENER_CONFIG.items():

        def create_listener(field: gr.Textbox, config: TextFieldListenerConfig):
            @field.change(inputs=[field, state], outputs=[state])
            def update_field(entered_value: str, state: AppState):
                if config["parse_as_tuple"]:
                    value = tuple(entered_value.replace(" ", "").split(","))
                else:
                    value = entered_value
                return replace(state, **{config["state_field"]: value})

            return update_field

        create_listener(input_element, config)

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
                state.dataset_name,
                splits=list(state.dataset_splits),
                version=None if not state.dataset_version else state.dataset_version,
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
