"""Configurable judge messages and optional screening, without implicit prompt text."""

from collections.abc import Callable
from dataclasses import dataclass
import json
import re
import unicodedata
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
)

from evalsense.utils.text import format_template

type JudgeInputScreen = Callable[[str], bool]
type JudgeRequest = str | JudgePrompt

PROTOCOL = "evalsense.judge-input.v2"

_INSTRUCTION_PATTERNS = re.compile(
    r"\b(?:ignore|disregard|override)\s+(?:all\s+)?"
    r"(?:(?:previous|prior|above|system|developer|evaluation)\s+){0,2}"
    r"(?:instructions|rubric|rules)\b"
    r"|\b(?:assign|give|return|output)\s+(?:me\s+|a\s+|the\s+)?"
    r"(?:maximum|highest|perfect|full)\s+(?:score|rating|marks)\b"
    r"|(?:^|\n)\s*(?:system|developer)\s*:"
    r"|<\|(?:im_start|start_header_id)\|>\s*(?:system|developer)",
    re.IGNORECASE,
)


class JudgeInputError(ValueError):
    """The configured prompt or input screen rejected a judge request."""


class JudgeOutputError(ValueError):
    """The judge did not return a valid response for scoring."""


def screen_instruction_phrases(text: str) -> bool:
    """Optional English-language screen: return True to allow the input.

    This heuristic can flag quoted instructions and miss paraphrases, encodings
    or other languages. It never changes the text and is not enabled by default.
    """
    return _INSTRUCTION_PATTERNS.search(unicodedata.normalize("NFKC", text)) is None


@dataclass(frozen=True)
class JudgePrompt:
    """The complete, caller-defined system and user messages for a judge.

    No guard, rubric, output instruction or delimiter is added automatically.
    Keep the system message trusted; put evaluated and model-generated text
    only in the user message. A caller that inserts evidence into the system
    message defeats this boundary. Role separation is not an injection guarantee.
    """

    system: str
    user: str

    @classmethod
    def from_template(
        cls, system_prompt: str, user_template: str, **fields: Any
    ) -> "JudgePrompt":
        """Format only the user template; preserve the supplied system text."""
        return cls(system_prompt, format_template(user_template, **fields))

    @classmethod
    def from_data(cls, system_prompt: str, **fields: Any) -> "JudgePrompt":
        """Optionally encode evidence as JSON; the system text must describe it."""
        try:
            user = json.dumps(fields, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            raise JudgeInputError("Judge evidence is not JSON serialisable.") from None
        return cls(system_prompt, user)

    def messages(self) -> list[ChatMessage]:
        """Return exactly the two configured messages."""
        if not isinstance(self.system, str) or not isinstance(self.user, str):
            raise JudgeInputError("Judge system and user messages must be strings.")
        return [
            ChatMessageSystem(content=self.system),
            ChatMessageUser(content=self.user),
        ]


async def generate_judge(
    model: Model,
    prompt: JudgeRequest,
    config: GenerateConfig,
    *,
    input_screen: JudgeInputScreen | None = None,
) -> tuple[ModelOutput, str | list[dict[str, Any]]]:
    """Send the configured prompt without tools and record its exact content.

    A string preserves the legacy single-user-message API. The optional screen
    receives the complete rendered user message before each request; True allows,
    False rejects. The system message is not screened. No output or input is
    silently rewritten, and no automatic retry is performed.
    """
    request: str | list[ChatMessage]
    recorded: str | list[dict[str, Any]]
    if isinstance(prompt, JudgePrompt):
        request = prompt.messages()
        user_text = prompt.user
        recorded = [message.model_dump(mode="json") for message in request]
    elif isinstance(prompt, str):
        request = recorded = user_text = prompt
    else:
        raise JudgeInputError("Judge prompt must be a string or JudgePrompt.")

    if input_screen is not None:
        try:
            allowed = input_screen(user_text)
        except Exception:
            raise JudgeInputError("Judge input screening failed.") from None
        if allowed is not True:
            raise JudgeInputError("Judge input rejected by the configured screen.")

    output = await model.generate(request, config=config, tools=[], tool_choice="none")
    if getattr(output, "error", None) or len(output.choices) != 1:
        raise JudgeOutputError("Judge must return one successful response.")
    if any(choice.message.tool_calls for choice in output.choices):
        raise JudgeOutputError("Judge tool calls are not permitted.")
    return output, recorded


def validate_rating(output: ModelOutput, min_score: int, max_score: int) -> int:
    """Require one complete in-range integer before any probability weighting."""
    text = output.completion.strip()
    if re.fullmatch(r"[+-]?[0-9]+", text):
        try:
            rating = int(text)
        except ValueError:
            pass
        else:
            if min_score <= rating <= max_score:
                return rating
    raise JudgeOutputError("Judge must return one integer within the rating scale.")
