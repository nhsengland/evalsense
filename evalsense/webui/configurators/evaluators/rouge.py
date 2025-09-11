from typing import override

import gradio as gr

from evalsense.evaluation import Evaluator
from evalsense.evaluation.evaluators import get_rouge_evaluator
from evalsense.webui.configurators.evaluator_configurator import (
    ConfiguratorInput,
    EvaluatorConfigurator,
    evaluator_configurator,
)


@evaluator_configurator
class RougeConfigurator(EvaluatorConfigurator):
    """Configurator for the ROUGE evaluator."""

    name = "ROUGE"

    @override
    def input_widget(self) -> list[ConfiguratorInput]:
        """Constructs the input widget for ROUGE.

        Returns:
            list[ConfiguratorInput]: The input fields for the configurator widget.
        """
        return [
            {
                "input_name": "name",
                "component": gr.Textbox(
                    label="Metric Name",
                    info="The name of the metric to show in the results.",
                    value="ROUGE",
                ),
                "parser": None,
            },
        ]

    @override
    def instantiate_evaluator(self, **kwargs) -> Evaluator:
        """
        Instantiates the ROUGE evaluator according to the specified configuration.

        Args:
            **kwargs: The keyword arguments specifying evaluator configuration.

        Returns:
            Evaluator: The instantiated evaluator.
        """
        return get_rouge_evaluator(**kwargs)
