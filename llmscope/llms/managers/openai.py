from llmscope.constants import OPENAI_API_KEY
from openai import OpenAI

from llmscope.llms import LlmManager


class OpenAiLlmManager(LlmManager):
    """A manager for OpenAI models.

    Attributes:
        model_name (str): The OpenAI model name.
        client (OpenAI): The OpenAI client.
    """

    def __init__(
        self,
        model_name,
        api_key=None,
    ):
        """Initializes an LLM manager for OpenAI models.

        Args:
            model_name (str): The OpenAI model name.
            api_key (str, optional): The OpenAI API key. Defaults to OPENAI_API_KEY value.
        """
        if api_key is None:
            if OPENAI_API_KEY is not None:
                api_key = OPENAI_API_KEY
            else:
                raise ValueError("API key is required for OpenAI models.")
        if "openai:" in model_name:
            model_name = model_name.replace("openai:", "", 1)
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
        )

    def chat_completion(
        self,
        messages,
        print_output=False,
        seed=42,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.0,
        **kwargs,
    ):
        """Generates a chat completion for the given messages.

        Args:
            messages (dict): The chat messages to generate completion for.
            print_output (bool, optional): Whether to print the model output. Defaults to False.
            seed (int, optional): The random seed. Defaults to 42.
            max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            temperature (float, optional): The sampling temperature. Defaults to 0.7.
            top_p (float, optional): The nucleus sampling parameter. Defaults to 0.95.
            repetition_penalty (float, optional): The repetition penalty. Defaults to 1.0.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            (str): The generated chat completion.
        """
        for message in messages:
            if message["role"] == "system":
                message["role"] = "developer"

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            seed=seed,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            presence_penalty=repetition_penalty,
            **kwargs,
        )

        response = completion.choices[0].message.content

        if print_output:
            print(response, flush=True)

        return response
