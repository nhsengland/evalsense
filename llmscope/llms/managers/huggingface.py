import torch
import transformers
from transformers import BitsAndBytesConfig

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
            device_map="auto",
            model_kwargs={
                "torch_dtype": "auto",
                "quantization_config": quantization_config,
                "cache_dir": MODELS_PATH,
                **model_kwargs,
            },
        )

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
        """Generates a chat completion for the given messages.

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
        transformers.set_seed(seed)
        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            **kwargs,
        )[0]["generated_text"]

        if print_output:
            print(response, flush=True)

        if generated_text_only:
            response = response.replace(prompt, "", 1).strip()

        return response
