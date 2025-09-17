from typing import override

import gradio as gr

from evalsense.evaluation import Evaluator
from evalsense.evaluation.evaluators import get_bertscore_evaluator
from evalsense.webui.configurators.evaluator_configurator import (
    ConfiguratorInput,
    EvaluatorConfigurator,
    configurator,
)
from evalsense.webui.utils import empty_is_none_parser_for


@configurator
class BertScoreConfigurator(EvaluatorConfigurator):
    """Configurator for the BERTScore evaluator."""

    name = "BERTScore"

    @override
    def input_widget(self) -> list[ConfiguratorInput]:
        """Constructs the input widget for BERTScore.

        Returns:
            list[ConfiguratorInput]: The input fields for the configurator widget.
        """
        return [
            {
                "input_name": "name",
                "component": gr.Textbox(
                    label="Metric Name",
                    info="The name of the metric to show in the results.",
                    value="BERTScore",
                ),
                "parser": None,
            },
            {
                "input_name": "model_type",
                "component": gr.Textbox(
                    label="Model Type",
                    info="The type of BERT model to use. See [this Google sheet](https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/view) for available models.",
                    value="microsoft/deberta-xlarge-mnli",
                ),
                "parser": None,
            },
            {
                "input_name": "lang",
                "component": gr.Textbox(
                    label="Language",
                    info="The language of the text to evaluate.",
                    value="en",
                ),
                "parser": None,
            },
            {
                "input_name": "num_layers",
                "component": gr.Textbox(
                    label="Layer Number",
                    info="The layer of representations to use. When empty, defaults to the best layer according to the WMT16 correlation data.",
                    value="",
                ),
                "parser": empty_is_none_parser_for(int),
            },
            {
                "input_name": "device",
                "component": gr.Textbox(
                    label="Device",
                    info="The device to use for computing the contextual embeddings. If this argument is left empty, the model will be loaded on `cuda:0` if available.",
                    value="",
                ),
                "parser": empty_is_none_parser_for(str),
            },
        ]

    @override
    def instantiate_evaluator(self, **kwargs) -> Evaluator:
        """
        Instantiates the BERTScore evaluator according to the specified configuration.

        Args:
            **kwargs (dict): The keyword arguments specifying evaluator configuration.

        Returns:
            Evaluator: The instantiated evaluator.
        """
        return get_bertscore_evaluator(**kwargs)
