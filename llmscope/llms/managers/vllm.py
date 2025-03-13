from vllm import LLM, SamplingParams

from llmscope.constants import MODELS_PATH
from llmscope.llms import LlmManager


class VLlmManager(LlmManager):
    """A manager for vLLM models (typically much faster than HuggingFace models).

    Attributes:
        llm (LLM): The vLLM model.
    """

    def __init__(
        self,
        name: str,
        max_model_len: int = 16384,
        model_kwargs: dict = {},
    ):
        """
        Initializes an LLM manager for vLLM models.

        Args:
            name (str): The vLLM model name.
            max_model_len (int, optional): The maximum length of the model. Defaults to 16384.
            model_kwargs (dict, optional): Additional model keyword arguments. Defaults to {}.
        """
        super().__init__()
        self.name = name
        self.llm = LLM(
            name,
            max_model_len=max_model_len,
            download_dir=MODELS_PATH,
            **model_kwargs,
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
    ) -> str | list[str]:
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
            **kwargs (dict): Additional keyword arguments to sampling parameters.

        Returns:
            (str | list[str]): The generated chat completion(s).
        """
        sampling_params = SamplingParams(
            seed=seed,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
        responses = self.llm.chat(
            messages, sampling_params=sampling_params, use_tqdm=show_progress
        )

        if generated_text_only:
            responses = [r.outputs[0].text for r in responses]
        else:
            responses = [
                m + [{"role": "assistant", "content": r.outputs[0].text}]
                for m, r in zip(messages, responses)
            ]

        if len(responses) == 1:
            responses = responses[0]

        if print_output:
            print(responses, flush=True)

        return responses
