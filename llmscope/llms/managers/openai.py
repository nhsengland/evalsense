from llmscope.constants import OPENAI_API_KEY
from openai import OpenAI
from tenacity import Retrying, stop_after_attempt, wait_random_exponential
from tqdm.auto import tqdm

from llmscope.llms import LlmManager


class OpenAiLlmManager(LlmManager):
    """A manager for OpenAI models.

    Attributes:
        model_name (str): The OpenAI model name.
        client (OpenAI): The OpenAI client.
    """

    def __init__(
        self,
        name,
        api_key=None,
    ):
        """Initializes an LLM manager for OpenAI models.

        Args:
            name (str): The OpenAI model name.
            api_key (str, optional): The OpenAI API key. Defaults to OPENAI_API_KEY value.
        """
        if api_key is None:
            if OPENAI_API_KEY is not None:
                api_key = OPENAI_API_KEY
            else:
                raise ValueError("API key is required for OpenAI models.")
        if "openai:" in name:
            name = name.replace("openai:", "", 1)
        self.name = name
        self.client = OpenAI(
            api_key=api_key,
        )

    def chat_completion(
        self,
        messages: list,
        print_output: bool = False,
        seed: int = 42,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        show_progress: bool = True,
        generated_text_only: bool = True,
        **kwargs,
    ):
        """Generates a chat completion for the given messages.

        Args:
            messages (list): The chat conversation(s) to generate completion(s) for,
                in HuggingFace chat format. Can be a list of multiple conversations.
            print_output (bool, optional): Whether to print the model output. Defaults to False.
            seed (int, optional): The random seed. Defaults to 42.
            max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            temperature (float, optional): The sampling temperature. Defaults to 0.7.
            top_p (float, optional): Only the smallest set of most probable tokens with probabilities
                summing to `top_p` or higher are kept for generation. Defaults to 0.95.
            repetition_penalty (float, optional): The repetition penalty. Defaults to 1.0.
            show_progress (bool, optional): Whether to show progress. Defaults to True.
            generated_text_only (bool, optional): Whether to return only the generated text. Defaults to True.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            (str): The generated chat completion.
        """
        if isinstance(messages[0], dict):
            messages = [messages]

        for conversation in messages:
            for message in conversation:
                if message["role"] == "system":
                    message["role"] = "developer"

        completions = []
        for conversation in tqdm(
            messages, desc="Generating outputs", disable=not show_progress
        ):
            for attempt in Retrying(
                wait=wait_random_exponential(min=1, max=60),
                stop=stop_after_attempt(6),
                reraise=True,
            ):
                with attempt as _:
                    completion = self.client.chat.completions.create(
                        model=self.name,
                        messages=conversation,
                        seed=seed,
                        temperature=temperature,
                        max_tokens=max_new_tokens,
                        top_p=top_p,
                        presence_penalty=repetition_penalty,
                        **kwargs,
                    )
                    completions.append(completion)

        if generated_text_only:
            completions = [c.choices[0].message.content for c in completions]
        else:
            completions = [
                m + [{"role": "assistant", "content": c.choices[0].message.content}]
                for m, c in zip(messages, completions)
            ]

        if len(completions) == 1:
            completions = completions[0]

        if print_output:
            print(completions, flush=True)

        return completions
