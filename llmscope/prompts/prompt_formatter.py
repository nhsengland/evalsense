from typing import Protocol


class PromptFormattingFunction(Protocol):
    """A protocol for a function that formats prompts."""

    def __call__(self, **kwargs) -> list[dict]:
        """Formats inputs into a prompt.

        Args:
            **kwargs (dict): Arguments for constructing the prompt.

        Returns:
            (list[dict]): The formatted prompt in the HuggingFace chat messages format.
        """
        ...


class PromptFormatter:
    """A class for formatting prompts for a specific task."""

    name: str

    def __init__(
        self, name: str, formatting_function: PromptFormattingFunction
    ) -> None:
        """Initializes the prompt formatter.

        Args:
            name (str): The name of the prompt formatter.
            formatting_function (PromptFormattingFunction): The function used to
                format the prompt.
        """
        self.name = name
        self.formatting_function = formatting_function

    def __call__(self, **kwargs) -> list[dict]:
        """Formats inuts into a prompt.

        Args:
            **kwargs (dict): Arguments for constructing the prompt.

        Returns:
            (list[dict]): The formatted prompt in the HuggingFace chat messages format.
        """
        return self.formatting_function(**kwargs)
