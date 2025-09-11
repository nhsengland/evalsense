from typing import override

import gradio as gr
from inspect_ai.model import GenerateConfigArgs

from evalsense.evaluation import Evaluator
from evalsense.evaluation.evaluators import get_g_eval_evaluator
from evalsense.generation.model_config import ModelConfig
from evalsense.webui.configurators.evaluator_configurator import (
    ConfiguratorInput,
    EvaluatorConfigurator,
    evaluator_configurator,
)
from evalsense.webui.utils import dict_parser


@evaluator_configurator
class GEvalConfigurator(EvaluatorConfigurator):
    """Configurator for the G-Eval evaluator."""

    name = "G-Eval"

    @override
    def input_widget(self) -> list[ConfiguratorInput]:
        """Constructs the input widget for G-Eval.

        Returns:
            list[ConfiguratorInput]: The input fields for the configurator widget.
        """
        return [
            {
                "input_name": "name",
                "component": gr.Textbox(
                    label="Metric Name",
                    info="The name of the metric to show in the results.",
                    value="G-Eval",
                ),
                "parser": None,
            },
            {
                "input_name": "quality_name",
                "component": gr.Textbox(
                    label="Quality Name",
                    info="The name of the quality to be evaluated by G-Eval.",
                    value="Unknown",
                ),
                "parser": None,
            },
            {
                "input_name": "prompt_template",
                "component": gr.TextArea(
                    label="Prompt Template",
                    info="The prompt template to use for evaluation. The supplied template should be a Python f-string with `{prediction}` and (optionally) `{reference}` as placeholders, as well as any additional placeholders for entries in Inspect AI sample/task state metadata. The template should instruct the judge model to respond with a numerical score between the specified `min_score` and `max_score`.",
                    max_lines=15,
                ),
                "parser": None,
            },
            {
                "input_name": "model_name",
                "component": gr.Textbox(
                    label="Model Name",
                    info="The name of the model to use as a judge following the [Inspect AI naming conventions](https://inspect.aisi.org.uk/models.html).",
                ),
                "parser": None,
            },
            {
                "input_name": "model_args",
                "component": gr.Textbox(
                    label="Model Arguments",
                    info="The arguments to pass to the model during evaluation, formatted as a Python dictionary. These will be passed to the [`get_model`](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#get_model) function when creating the model.",
                ),
                "parser": dict_parser,
            },
            {
                "input_name": "generation_args",
                "component": gr.Textbox(
                    label="Generation Arguments",
                    info="The arguments to pass to the model during generation, formatted as a Python dictionary. See [`GenerateConfigArgs`](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfigargs) Inspect AI documentation for valid values.",
                ),
                "parser": dict_parser,
            },
            {
                "input_name": "min_score",
                "component": gr.Number(
                    label="Min Score",
                    info="The minimum score on the G-Eval rating scale.",
                    value=1,
                ),
                "parser": int,
            },
            {
                "input_name": "max_score",
                "component": gr.Number(
                    label="Max Score",
                    info="The maximum score on the G-Eval rating scale.",
                    value=5,
                ),
                "parser": int,
            },
            {
                "input_name": "logprobs",
                "component": gr.Checkbox(
                    label="Log Probs",
                    info="Whether to use log probabilities of the generated tokens to compute a weighted evaluation score.",
                    value=True,
                ),
                "parser": None,
            },
            {
                "input_name": "top_logprobs",
                "component": gr.Number(
                    label="Top Log Probs",
                    info="The number of top log probabilities to consider for each generated token.",
                    value=20,
                ),
                "parser": int,
            },
            {
                "input_name": "normalise",
                "component": gr.Checkbox(
                    label="Normalise",
                    info="Whether to normalise the evaluation scores to be between 0 and 1.",
                    value=True,
                ),
                "parser": None,
            },
        ]

    @override
    def instantiate_evaluator(self, **kwargs) -> Evaluator:
        """
        Instantiates the BERTScore evaluator according to the specified configuration.

        Args:
            **kwargs: The keyword arguments specifying evaluator configuration.

        Returns:
            Evaluator: The instantiated evaluator.
        """
        model_name = kwargs.pop("model_name")
        model_args = kwargs.pop("model_args", {})
        generation_args = kwargs.pop("generation_args", {})
        model_config = ModelConfig(
            model=model_name,
            model_args=model_args,
            generation_args=GenerateConfigArgs(**generation_args),
        )

        return get_g_eval_evaluator(**kwargs, model_config=model_config)
