from abc import abstractmethod
from typing import Callable, Protocol, Type, TypedDict

from gradio.blocks import Block

from evalsense.evaluation.evaluator import Evaluator


class EvaluatorConfiguratorRegistry:
    """A registry for evaluator configurators."""

    registry: dict[str, Type["EvaluatorConfigurator"]] = {}

    @classmethod
    def register(cls, configurator: Type["EvaluatorConfigurator"]):
        """Register a new evaluator configurator.

        Args:
            configurator (Type["EvaluatorConfigurator"]): The evaluator configurator to register.
        """
        cls.registry[configurator.name] = configurator

    @classmethod
    def get(cls, name: str) -> Type["EvaluatorConfigurator"]:
        """Get an evaluator configurator by name.

        Args:
            name (str): The name of the evaluator configurator to retrieve.

        Returns:
            Type["EvaluatorConfigurator"]: The requested evaluator configurator.
        """
        if name not in cls.registry:
            raise ValueError(f"No configurator for {name} has been registered.")
        return cls.registry[name]


def configurator(
    configurator: Type["EvaluatorConfigurator"],
) -> Type["EvaluatorConfigurator"]:
    """Decorator to register an evaluator configurator.

    Args:
        configurator (Type["EvaluatorConfigurator"]): The evaluator configurator to register.

    Returns:
        Type["EvaluatorConfigurator"]: The registered evaluator configurator.
    """
    EvaluatorConfiguratorRegistry.register(configurator)
    return configurator


class ConfiguratorInput(TypedDict):
    """A typed dictionary for the input fields of the configurator widget.

    Attributes:
        input_name (str): The name of the input field.
        component (Block): The Gradio component for the input field.
        parser (Callable): A callable to parse the input value.
    """

    input_name: str
    component: Block
    parser: Callable | None


class EvaluatorConfigurator(Protocol):
    """A protocol for configuring evaluators.

    Attributes:
        name (str): The string ID of the evaluator. A class attribute.
    """

    name: str

    @classmethod
    def create(cls, name: str) -> "EvaluatorConfigurator":
        """Create a configurator for the specified evaluator.

        Args:
            name (str): The name of the evaluator for which the configurator
                should be created.

        Returns:
            EvaluatorConfigurator: The created evaluator configurator instance.
        """
        configurator = EvaluatorConfiguratorRegistry.get(name)
        return configurator()

    @abstractmethod
    def input_widget(self) -> list[ConfiguratorInput]:
        """Constructs the configurator widget.

        Returns:
            list[ConfiguratorInput]: The input fields for the configurator widget.
        """
        ...

    @abstractmethod
    def instantiate_evaluator(self, **kwargs) -> Evaluator:
        """
        Instantiates the evaluator according to the specified configuration.

        Args:
            **kwargs (dict): The keyword arguments specifying evaluator configuration.

        Returns:
            Evaluator: The instantiated evaluator.
        """
        ...
