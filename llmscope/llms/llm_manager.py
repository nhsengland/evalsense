from abc import ABC, abstractmethod


class LlmManager(ABC):
    """
    An abstract class for LLM manager objects.
    """

    name: str

    @abstractmethod
    def chat_completion(
        self,
        messages: dict,
        print_output: bool = False,
        seed: int = 42,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        generated_text_only=True,
        **kwargs,
    ) -> str:
        """Generates a chat completion for the given message.

        Args:
            messages (dict): The chat messages to generate completion for.
            print_output (bool, optional): Whether to print the model output. Defaults to False.
            seed (int, optional): The random seed. Defaults to 42.
            max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            do_sample (bool, optional): Whether to sample the output. Defaults to True.
            temperature (float, optional): The sampling temperature. Defaults to 0.7.
            top_p (float, optional): The nucleus sampling parameter. Defaults to 0.95.
            repetition_penalty (float, optional): The repetition penalty. Defaults to 1.0.
            generated_text_only (bool, optional): Whether to return only the generated text. Defaults to True.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            (str): The generated chat completion.
        """
        pass

    def __call__(self, *args, **kwargs) -> str:
        """Generates a chat completion for the given message.

        Args:
            *args (dict): Positional arguments (see `chat_completion`).
            **kwargs (dict): Keyword arguments (see `chat_completion`).
        Returns:
            (str): The generated chat completion.
        """
        return self.chat_completion(*args, **kwargs)
