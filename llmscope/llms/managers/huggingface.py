from datasets import Dataset
import torch
import transformers
from transformers import BitsAndBytesConfig
from tqdm.auto import tqdm

from llmscope.constants import MODELS_PATH
from llmscope.llms import LlmManager


class HuggingFaceLlmManager(LlmManager):
    """A manager for HuggingFace models.

    Attributes:
        pipeline (transformers.Pipeline): The HuggingFace model pipeline.
    """

    def __init__(
        self,
        model_name: str,
        quantization: str = "none",
        device_map: str = "auto",
        model_kwargs: dict = {},
    ):
        """
        Initializes an LLM manager for HuggingFace models.

        Args:
            model_name (str): The HuggingFace model name.
            quantization (str, optional): The quantization method to use. Defaults to "none".
            model_kwargs (dict, optional): Additional model keyword arguments. Defaults to {}.
        """
        super().__init__()
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantization == "none":
            quantization_config = None
        else:
            raise ValueError(f"Invalid quantization value {quantization}")

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            device_map=device_map,
            model_kwargs={
                "torch_dtype": "auto",
                "quantization_config": quantization_config,
                "cache_dir": MODELS_PATH,
                **model_kwargs,
            },
        )

    def chat_completion(
        self,
        messages: list,
        print_output: bool = False,
        seed: int = 42,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        show_progress: bool = True,
        generated_text_only=True,
        **kwargs,
    ) -> str | list[str]:
        """Generates a chat completion for the given messages.

        Args:
            messages (list): The chat conversation(s) to generate completion(s) for,
                in HuggingFace chat format. Can be a list of multiple conversations.
            print_output (bool, optional): Whether to print the model output. Defaults to False.
            seed (int, optional): The random seed. Defaults to 42.
            max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            do_sample (bool, optional): Whether to sample the output. Defaults to True.
            temperature (float, optional): The sampling temperature. Defaults to 0.7.
            top_p (float, optional): Only the smallest set of most probable tokens with probabilities
                summing to `top_p` or higher are kept for generation. Defaults to 0.95.
            repetition_penalty (float, optional): The repetition penalty. Defaults to 1.0.
            show_progress (bool, optional): Whether to show progress. Defaults to True.
            generated_text_only (bool, optional): Whether to return only the generated text. Defaults to True.
            **kwargs (dict): Additional keyword arguments to generation.

        Returns:
            (str | list[str]): The generated chat completion(s).
        """
        transformers.set_seed(seed)

        if isinstance(messages[0], dict):
            messages = [messages]

        responses = []
        for conversation in tqdm(
            messages, desc="Generating outputs", disable=not show_progress
        ):
            response = self.pipeline(
                conversation,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                **kwargs,
            )
            responses.append(response)

        responses = [r[0]["generated_text"] for r in responses]
        if generated_text_only:
            responses = [r[-1]["content"] for r in responses]

        if len(responses) == 1:
            responses = responses[0]

        if print_output:
            print(responses, flush=True)

        return responses
