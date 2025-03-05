from typing import Protocol


class PromptFormatter(Protocol):
    """A protocol for formatting prompts for a specific task."""

    def __call__(self, **kwargs) -> dict:
        """Formats inuts into a prompt.

        Args:
            **kwargs (dict): Arguments for constructing the prompt.

        Returns:
            (dict): The formatted prompt in the HuggingFace chat messages format.
        """
        ...
