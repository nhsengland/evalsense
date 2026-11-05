import gradio as gr

from evalsense.webui.configurators.evaluator_configurator import (
    EvaluatorConfigurator,
    EvaluatorConfiguratorRegistry,
)
from evalsense.webui.state import AppState
from evalsense.webui.utils import configuration_for_display
from evalsense.utils.credentials import validate_public_config


def evaluators_tab(state: gr.State):
    """Renders the evaluators tab user interface.

    Arguments:
        state (gr.State): The current state of the Gradio application.
    """
    # Evaluators tab user interface
    gr.Markdown("Use this tab to configure and manage evaluators.")
    gr.Markdown("## New Evaluator Configuration")
    evaluator_dropdown = gr.Dropdown(
        label="Select Evaluator",
        choices=list(EvaluatorConfiguratorRegistry.registry.keys()),
        value=None,
    )

    @gr.render(inputs=evaluator_dropdown)
    def show_configurator_widget(dropdown_value: str):
        if dropdown_value is None:
            gr.Markdown("### No Evaluator Selected")
            return

        gr.Markdown(f"### {dropdown_value} Configuration")
        configurator = EvaluatorConfigurator.create(dropdown_value)
        configurator_inputs = configurator.input_widget()
        evaluator_add_button = gr.Button("Add Evaluator", variant="primary")

        @evaluator_add_button.click(
            inputs=[ci["component"] for ci in configurator_inputs] + [state],
            outputs=[evaluator_dropdown, state],
        )
        def add_evaluator(*args):
            # Process inputs
            evaluator_args = args[:-1]
            parsed_args = {}
            for configurator_input, arg in zip(
                configurator_inputs, evaluator_args, strict=True
            ):
                parser = configurator_input["parser"]
                input_name = configurator_input["input_name"]
                value = arg
                if parser:
                    try:
                        value = parser(arg)
                    except Exception:
                        raise gr.Error(
                            f"Invalid {input_name}. Check the format and keep credentials out of configuration."
                        ) from None
                parsed_args[input_name] = value

            # Validate before retaining options in session state.
            validate_public_config(parsed_args)
            # Update state
            state = args[-1]
            state["evaluator_configs"].append(
                {"evaluator_name": dropdown_value, "evaluator_args": parsed_args}
            )
            return gr.update(value=None), state

    @gr.render(inputs=state)
    def show_evaluator_configs(local_state: AppState):
        gr.Markdown("## Current Evaluator Configurations")

        if not local_state["evaluator_configs"]:
            gr.Markdown("No evaluators configured yet.")

        for i, evaluator_config in enumerate(local_state["evaluator_configs"]):
            gr.Markdown(
                f"### Evaluator #{i + 1} — {evaluator_config['evaluator_name']}"
            )
            gr.JSON(
                configuration_for_display(dict(evaluator_config["evaluator_args"])),
                label="Evaluator Config",
            )
            evaluator_remove_button = gr.Button("Remove Evaluator", variant="stop")

            @evaluator_remove_button.click(inputs=[state], outputs=[state])
            def remove_evaluator(state: AppState, idx=i):
                new_evaluator_configs = list(state["evaluator_configs"])
                new_evaluator_configs.pop(idx)
                state["evaluator_configs"] = new_evaluator_configs
                return state
