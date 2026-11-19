from collections import defaultdict
from dataclasses import dataclass
import math
import regex
from typing import Callable, Literal, overload

from inspect_ai.model import ModelOutput
import numpy as np


def format_template(template: str, **kwargs) -> str:
    """
    Format a template string with the provided keyword arguments.

    Args:
        template (str): The template string to format.
        **kwargs (dict[str, any]): Keyword arguments to replace placeholders in the template.

    Returns:
        str: The formatted string.
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise KeyError(
            f"Missing key in template formatting: {str(e)}. "
            f"The provided keys were: {list(kwargs.keys())}. "
            "Ensure all placeholders in the template are provided in kwargs."
        )


def extract_lines(
    text: str,
    include_filter_fun: Callable[[str], bool] = lambda _: True,
    trim_lines: bool = True,
) -> list[str]:
    """
    Extract lines from the text based on a filter function.

    Args:
        text (str): The text to extract lines from.
        include_filter_fun (Callable[[str], bool], optional): A function that
            takes a line and returns True if it should be included, False
            otherwise. Defaults to a function that includes all lines.
        trim_lines (bool, optional): Whether to trim bullet points, list numbers
            and whitespace from the beginning/end of each line. Defaults to True.

    Returns:
        list[str]: A list of extracted lines.
    """
    lines = text.splitlines()
    if trim_lines:
        lines = [regex.sub(r"^\s*\d+\.\s*", "", line) for line in lines]
        lines = [line.strip().lstrip("*").lstrip("-").strip() for line in lines]

    return [line for line in lines if include_filter_fun(line)]


@overload
def extract_ternary_answer(
    text: str, binary_only: Literal[True], unknown_on_mismatch: Literal[False] = False
) -> bool: ...
@overload
def extract_ternary_answer(
    text: str, binary_only: Literal[False], unknown_on_mismatch: bool = True
) -> bool | None: ...
def extract_ternary_answer(
    text: str, binary_only: bool, unknown_on_mismatch: bool = True
) -> bool | None:
    """
    Extract a ternary answer (True/False/Unknown) from the text.

    Valid answers are 'yes', 'no', 'true', 'false', 'unknown', and 'I don't know'.

    Args:
        text (str): The text to extract the answer from.
        binary_only (bool): If True, only 'yes' or 'no' are valid answers.
        unknown_on_mismatch (bool): If True, return None for answers not
            matching any of the valid answers. If False, raise an error.
            Only relevant if binary_only is False. Defaults to True.

    Returns:
        bool | None: The extracted answer - bool for True/False, None for Unknown.
    """
    pattern = r"\b(?:yes|no|true|false|unknown|i don't know)\b"
    match = regex.search(pattern, text, regex.IGNORECASE)
    if match:
        answer = match.group(0).lower()
        if answer in ["yes", "true"]:
            return True
        elif answer in ["no", "false"]:
            return False
        else:
            if binary_only:
                raise ValueError(
                    "Binary answer expected, but 'unknown' or 'I don't know' found."
                )
            return None
    if not binary_only and unknown_on_mismatch:
        return None
    raise ValueError(f"Unable to extract a binary answer from text: {text}.")


def validate_score_range(
    min_score: int, max_score: int, *, normalise: bool = False
) -> None:
    """Require non-negative integer endpoints and a usable normalisation span."""
    if type(min_score) is not int or type(max_score) is not int:
        raise ValueError(
            "Rating scale endpoints must be integers, not booleans or floats."
        )
    if min_score < 0 or max_score < min_score:
        raise ValueError("Rating scale must satisfy 0 <= min_score <= max_score.")
    if type(normalise) is not bool:
        raise ValueError("normalise must be a boolean.")
    if normalise and min_score == max_score:
        raise ValueError("Normalised ratings require min_score < max_score.")


def extract_score(text: str, min_score: int = 1, max_score: int = 10) -> int:
    """Parse one complete, in-range, non-negative decimal integer.

    Outer whitespace and leading zeroes are accepted. Signs, fractions, decimals,
    wrappers and explanatory text are rejected instead of searching for digits.

    Args:
        text (str): The complete judge response.
        min_score (int): The minimum valid score. Defaults to 1.
        max_score (int): The maximum valid score. Defaults to 10.

    Returns:
        int: The validated integer rating.

    Raises:
        ValueError: If the response is not a complete in-range integer.
    """
    validate_score_range(min_score, max_score)
    if not isinstance(text, str) or not regex.fullmatch(r"[0-9]+", text.strip()):
        raise ValueError("Expected one complete integer score without extra text.")
    try:
        score = int(text.strip())
    except ValueError:
        raise ValueError("Integer score exceeds the supported range.") from None
    if not min_score <= score <= max_score:
        raise ValueError("Integer score is outside the configured rating scale.")
    return score


def _eval_weighted_options[T](
    output: ModelOutput,
    normalise_token: Callable[[str], str],
    matches_target_token: Callable[[str], bool],
    token_parsing_function: Callable[[str], T],
) -> dict[T, float]:
    """
    Extract weighted options from the model output.

    Args:
        output (ModelOutput): The model output containing logprobs.
        normalise_token (Callable[[str], str]): Function to normalise tokens.
        matches_target_token (Callable[[str], bool]): Function to check if a token
            matches the target (i.e., is at the position where we expect the answer).
        token_parsing_function (Callable[[str], T]): Function to parse the token
            into the desired option type. For invalid tokens that should be ignored,
            the function should raise a ValueError.

    Returns:
        dict[T, float]: A dictionary mapping each valid option to its probability.
    """
    if output.choices[0].logprobs is None:
        raise ValueError(
            "Logprobs for computing weighted options are not available. "
            + "Check if your model provider supports logprobs and disable "
            + "using weighted options if needed by setting logprobs=False."
        )

    # First, identify matching target token
    logprobs = output.choices[0].logprobs.content
    target_logprobs = None
    for logprob in logprobs:
        token_text = normalise_token(logprob.token)
        if matches_target_token(token_text):
            target_logprobs = logprob
            break
    if target_logprobs is None:
        raise ValueError(
            "Unable to identify correct token for computing weighted options."
        )
    if target_logprobs.top_logprobs is None:
        raise ValueError(
            "Top logprobs are not available for computing weighted options."
        )

    # Then, calculate the probability for each candidate output
    probabilities: dict[T, float] = defaultdict(float)
    total_probability = 0.0
    for logprob in target_logprobs.top_logprobs:
        clean_token = normalise_token(logprob.token)

        try:
            parsed_option = token_parsing_function(clean_token)
        except ValueError:
            # Ignore invalid options
            continue

        score_probability = np.exp(logprob.logprob)
        probabilities[parsed_option] += score_probability
        total_probability += score_probability

    if not 0 < total_probability <= 1 + 1e-5:
        raise ValueError(
            "Valid answer probabilities must have positive finite mass no greater than one."
        )

    # Normalise the option probabilities
    for option in probabilities:
        probabilities[option] /= total_probability

    return probabilities


def extract_weighted_binary_answer(output: ModelOutput) -> float:
    """
    Extract a weighted binary answer from the model output.

    Args:
        output (ModelOutput): The model output containing logprobs.

    Returns:
        float: The model probability of the answer being True.
    """
    valid_options = ["true", "false", "yes", "no"]

    def normalise_token(token: str) -> str:
        return token.strip().replace("▁", "").replace("_", "").lower()

    def match_target_token(token: str) -> bool:
        return token in valid_options

    def token_parsing_function(token: str) -> bool:
        if token not in valid_options:
            raise ValueError

        return token in ["true", "yes"]

    probabilities = _eval_weighted_options(
        output,
        normalise_token,
        match_target_token,
        token_parsing_function,
    )

    return probabilities[True]


@dataclass(frozen=True)
class WeightedRating:
    """A conditional mean and the returned rating probabilities that support it."""

    score: float
    probabilities: dict[int, float]
    probability_mass: float


def extract_weighted_rating(
    output: ModelOutput, min_score: int = 1, max_score: int = 10
) -> WeightedRating:
    """Average complete rating tokens from a one-token judge request.

    Every allowed rating must occur in the returned alternatives. The result is
    conditional on the valid rating tokens in that table, whose probabilities
    are renormalised. It is not a full-sequence probability estimate.

    Callers must request at most one output token. Missing ratings, multi-token
    responses and inconsistent probability evidence raise ValueError.
    """
    if len(output.choices) != 1:
        raise ValueError("Weighted ratings require exactly one judge response.")
    target_score = extract_score(output.completion, min_score, max_score)
    if output.completion.strip() != str(target_score):
        raise ValueError(
            "Weighted ratings require canonical integer labels without leading zeroes."
        )
    logprobs = output.choices[0].logprobs
    if logprobs is None or len(logprobs.content) != 1:
        raise ValueError(
            "Weighted ratings require exactly one output token with logprobs. "
            "Use logprobs=False for an explicit sampled rating."
        )
    token = logprobs.content[0]
    if not math.isfinite(token.logprob) or token.logprob > 0:
        raise ValueError(
            "The sampled rating token must have a finite non-positive log probability."
        )

    def rating_label(text: str) -> int:
        label = text.strip().removeprefix("▁").strip()
        value = extract_score(label, min_score, max_score)
        if label != str(value):
            raise ValueError(
                "Weighted rating tokens must use canonical integer labels."
            )
        return value

    try:
        matches_response = rating_label(token.token) == target_score
    except ValueError:
        matches_response = False
    if not matches_response:
        raise ValueError(
            "The single weighted rating token does not match the complete response."
        )
    if not token.top_logprobs:
        raise ValueError("Top logprobs are required for weighted ratings.")

    probabilities: dict[int, float] = defaultdict(float)
    seen: set[str] = set()
    sampled_logprob: float | None = None
    total_mass = 0.0
    for alternative in token.top_logprobs:
        if alternative.token in seen:
            raise ValueError("Weighted rating alternatives contain a duplicate token.")
        seen.add(alternative.token)
        logprob = alternative.logprob
        if math.isnan(logprob) or logprob > 0:
            raise ValueError(
                "Weighted rating log probabilities must be non-positive and not NaN."
            )
        mass = math.exp(logprob)
        total_mass += mass
        if alternative.token == token.token:
            sampled_logprob = logprob
        try:
            rating = rating_label(alternative.token)
        except ValueError:
            continue
        probabilities[rating] += mass

    if sampled_logprob is None or not math.isclose(
        sampled_logprob, token.logprob, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError(
            "The sampled token must have matching logprobs in the alternatives."
        )
    if total_mass > 1 + 1e-5:
        raise ValueError(
            "Reported token probabilities must have mass no greater than one."
        )
    if len(probabilities) != max_score - min_score + 1:
        missing = []
        for rating in range(min_score, max_score + 1):
            if rating not in probabilities:
                missing.append(str(rating))
                if len(missing) == 5:
                    break
        raise ValueError(
            "Weighted ratings require a complete single-token alternative for every "
            f"rating on the scale. Missing ratings include {', '.join(missing)}. "
            "Multi-token labels and incomplete top_logprobs are unsupported. "
            "Increase top_logprobs where supported, or use logprobs=False "
            "for an explicit sampled rating."
        )
    rating_mass = math.fsum(probabilities.values())
    if not 0 < rating_mass <= 1 + 1e-5:
        raise ValueError(
            "Valid rating probabilities must have positive finite mass no greater than one."
        )
    normalised = {rating: mass / rating_mass for rating, mass in probabilities.items()}
    return WeightedRating(
        score=math.fsum(
            rating * probability for rating, probability in normalised.items()
        ),
        probabilities=normalised,
        probability_mass=rating_mass,
    )


def extract_weighted_score(
    output: ModelOutput, min_score: int = 1, max_score: int = 10
) -> float:
    """Return the conditional rating mean from validated one-token evidence.

    See extract_weighted_rating for the token and probability requirements.
    """
    return extract_weighted_rating(output, min_score, max_score).score
