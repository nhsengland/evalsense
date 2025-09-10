import gradio as gr

from evalsense.webui.state import AppModelConfig, AppState
from evalsense.webui.components.utils import dict_parser


def models_tab(state: gr.State):
    # Models tab user interface
    gr.Markdown("Use this tab to configure the models to evaluate.")
    gr.Markdown("## New Model Configuration")
    model_name_input = gr.Textbox(
        label="Model Name",
        info="The name of the model to evaluate following the [Inspect AI naming conventions](https://inspect.aisi.org.uk/models.html).",
    )
    model_args = gr.Textbox(
        label="Model Arguments",
        info="The arguments to pass to the model during evaluation, formatted as a Python dictionary. These will be passed to the [`get_model`](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#get_model) function when creating the model.",
    )
    generation_args = gr.Textbox(
        label="Generation Arguments",
        info="The arguments to pass to the model during generation, formatted as a Python dictionary. See [`GenerateConfigArgs`](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfigargs) Inspect AI documentation for valid values.",
    )
    model_add_button = gr.Button("Add Model", variant="primary")

    @model_add_button.click(
        inputs=[model_name_input, model_args, generation_args, state],
        outputs=[model_name_input, model_args, generation_args, state],
    )
    def add_model(name: str, args: str, gen_args: str, state: AppState):
        new_model: AppModelConfig = {
            "model_name": name,
            "model_args": dict_parser(args),
            "generation_args": dict_parser(gen_args),
        }
        state["model_configs"].append(new_model)
        return gr.update(value=""), gr.update(value=""), gr.update(value=""), state

    @gr.render(inputs=state)
    def show_model_configs(local_state: AppState):
        gr.Markdown("## Current Model Configurations")

        if not local_state["model_configs"]:
            gr.Markdown("No models configured yet.")

        for i, model_config in enumerate(local_state["model_configs"]):
            gr.Markdown(f"### Model #{i + 1} — {model_config['model_name']}")
            gr.JSON(dict(model_config), label="Model Config")
            model_remove_button = gr.Button("Remove Model", variant="stop")

            @model_remove_button.click(inputs=[state], outputs=[state])
            def remove_model(state: AppState, idx=i):
                new_model_configs = list(state["model_configs"])
                new_model_configs.pop(idx)
                state["model_configs"] = new_model_configs
                return state
