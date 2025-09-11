from typing import override

import gradio as gr

from evalsense.evaluation import Evaluator
from evalsense.evaluation.evaluators import get_bleu_evaluator
from evalsense.webui.configurators.evaluator_configurator import (
    ConfiguratorInput,
    EvaluatorConfigurator,
    evaluator_configurator,
)


@evaluator_configurator
class BleuConfigurator(EvaluatorConfigurator):
    """Configurator for the BLEU evaluator."""

    name = "BLEU"

    @override
    def input_widget(self) -> list[ConfiguratorInput]:
        """Constructs the input widget for BLEU.

        Returns:
            list[ConfiguratorInput]: The input fields for the configurator widget.
        """
        return [
            {
                "input_name": "name",
                "component": gr.Textbox(
                    label="Metric Name",
                    info="The name of the metric to show in the results.",
                    value="BLEU",
                ),
                "parser": None,
            },
            {
                "input_name": "scorer_name",
                "component": gr.Textbox(
                    label="Scorer Name",
                    info="The name of the internal scorer to show in the results.",
                    value="BLEU Precision",
                ),
                "parser": None,
            },
        ]

    @override
    def instantiate_evaluator(self, **kwargs) -> Evaluator:
        """
        Instantiates the BLEU evaluator according to the specified configuration.

        Args:
            **kwargs: The keyword arguments specifying evaluator configuration.

        Returns:
            Evaluator: The instantiated evaluator.
        """
        return get_bleu_evaluator(**kwargs)
