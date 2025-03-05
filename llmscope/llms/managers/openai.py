from llmscope.constants import OPENAI_API_KEY
from openai import OpenAI

from llmscope.llms import LlmManager


class OpenAiLlmManager(LlmManager):
    def __init__(
        self,
        model_name,
        api_key=None,
    ):
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
        print_result=False,
        seed=42,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.0,
        **kwargs,
    ):
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

        if print_result:
            print(response, flush=True)

        return response
