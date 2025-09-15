import gradio as gr

from evalsense.webui.state import AppState
from evalsense.webui.utils import (
    GradioInput,
    ListenerConfig,
    setup_listeners,
)


# TODO: Investigate better strategies for conditional rendering and managing state
#       Ideally, the whole tab would be rendered with state as an input, but it seems
#       gradio doesn't support this when state is being updated by action listeners
#       defined within the render function.
#       See also https://github.com/gradio-app/gradio/issues/9595.
def generation_tab(state: gr.State):
    """Renders the generation tab user interface.

    Arguments:
        state (gr.State): The current state of the Gradio application.
    """
    # Generation tab user interface
    gr.Markdown("Use this tab to configure the prompt to use during the generation.")
    gr.Markdown("## Generation Configuration")
    is_meta_eval_input = gr.Radio(
        label="Generation Type",
        info="What type of generation do you wish to perform? Standard generation involves generating new content based on a given prompt, while perturbation for meta-evaluation involves applying progressively aggressive output modifications to assess the effectiveness of different evaluation methods.",
        value=state.value["is_meta_eval"],
        choices=[
            ("Standard Generation", False),
            ("Perturbation for Meta-Evaluation", True),
        ],
    )

    generation_steps_name_input = gr.Textbox(
        label="Generation Steps Name",
        value=state.value["generation_steps_name"],
        info="The name of the used generation strategy.",
    )

    num_perturbation_tiers = gr.Number(
        label="Number of Perturbation Tiers",
        info="The number of perturbation tiers to use. Tiers with higher ID numbers should apply progressively more aggressive perturbations to the model outputs.",
        value=state.value["perturbation_tiers"],
        precision=0,
        minimum=2,
        visible=False,
    )

    @is_meta_eval_input.change(
        inputs=[is_meta_eval_input, state],
        outputs=[num_perturbation_tiers, state],
    )
    def update_is_meta_eval(is_meta: bool, state: AppState):
        state["is_meta_eval"] = is_meta
        # Reset inputs
        state["perturbation_tiers"] = 2
        state["perturbation_tier_subprompts"] = []
        return gr.update(visible=is_meta, value=2), state

    @num_perturbation_tiers.change(
        inputs=[num_perturbation_tiers, state],
        outputs=[state],
    )
    def update_perturbation_tiers(num_tiers: int, state: AppState):
        state["perturbation_tiers"] = num_tiers
        # Ensure the perturbation_tier_subprompts list has the correct length
        current_length = len(state["perturbation_tier_subprompts"])
        if current_length < num_tiers:
            state["perturbation_tier_subprompts"].extend(
                [""] * (num_tiers - current_length)
            )
        elif current_length > num_tiers:
            state["perturbation_tier_subprompts"] = state[
                "perturbation_tier_subprompts"
            ][:num_tiers]
        return state

    @gr.render(inputs=[is_meta_eval_input, num_perturbation_tiers])
    def show_perturbation_inputs(is_meta_eval: bool, perturbation_tiers: int):
        if is_meta_eval:
            gr.Markdown("## Perturbation Sub-Prompt Configuration")
            gr.Markdown(
                "In this section, you can provide subprompts specific to each perturbation tier. Each subprompt will be substituted into the `{perturbation_tier_subprompt}` placeholder in the system and user prompts below. This lets you share common prompt elements across tiers or supply completely different instructions for each tier. In the latter case, the user prompt may contain only the `{perturbation_tier_subprompt}` placeholder. Please double-check that the system or user prompt below actually includes the placeholder in verbatim. Otherwise, the subprompts defined here will not be used. Note that the perturbation tiers with higher ID numbers should apply progressively more aggressive perturbations, degrading the output quality with respect to the considered quality criterion."
            )
            for i in range(perturbation_tiers):
                tier_prompt_input = gr.TextArea(
                    label=f"Perturbation Tier #{i + 1} Sub-Prompt",
                    key=f"perturbation_tier_subprompt_{i}",
                    info="The prompt template to use for this perturbation tier.",
                    max_lines=15,
                )

                @tier_prompt_input.input(
                    inputs=[tier_prompt_input, state], outputs=[state]
                )
                def update_tier_prompt(tier_prompt: str, state: AppState, idx=i):
                    state["perturbation_tier_subprompts"][idx] = tier_prompt
                    return state

    gr.Markdown("## Prompt Configuration")
    system_prompt_input = gr.TextArea(
        label="System Prompt",
        info="The prompt to use for the system message. You can use Python f-string format to substitute the main input into a `{prompt}` placeholder, as well as for definiting placeholders for any additional metadata fields specified on the data tab.",
        max_lines=15,
    )
    user_prompt_input = gr.TextArea(
        label="User Prompt",
        info="The prompt to use for the user message. You can use Python f-string format to substitute the main input into a `{prompt}` placeholder, as well as for definiting placeholders for any additional metadata fields specified on the data tab.",
        max_lines=15,
    )

    # Textbox listeners
    LISTENER_CONFIG: dict[GradioInput, ListenerConfig] = {
        generation_steps_name_input: {
            "state_field": "generation_steps_name",
            "parser": None,
        },
        system_prompt_input: {"state_field": "system_prompt", "parser": None},
        user_prompt_input: {"state_field": "user_prompt", "parser": None},
    }
    setup_listeners(LISTENER_CONFIG, state)
