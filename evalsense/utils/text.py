from collections import defaultdict
import regex
from typing import Callable, Literal, overload

from inspect_ai.model import ModelOutput
import numpy as np


def format_template(template: str, **kwargs) -> str:
    """
    Format a template string with the provided keyword arguments.

    Args:
        template (str): The template string to format.
        **kwargs: Keyword arguments to replace placeholders in the template.

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


def extract_score(text: str, min_score: int = 1, max_score: int = 10) -> int:
    """
    Extract the first numerical score from text that falls between min_score and max_score.

    Args:
        text (str): The text to extract the score from.
        min_score (int): The minimum valid score. Defaults to 1.
        max_score (int): The maximum valid score. Defaults to 10.

    Returns:
        float | None: The extracted score if found and valid, otherwise None.
    """
    pattern = r"\b\d+\b"
    matches = regex.findall(pattern, text)

    for match in matches:
        try:
            score = int(match)
            if min_score <= score <= max_score:
                return score
        except ValueError:
            continue

    raise ValueError(f"Unable to extract a valid score from text: {text}.")


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

    assert total_probability > 0, "Total probability should be greater than zero."
    # Allow for probability slightly exceeding 1 due to floating point errors
    assert total_probability <= 1 + 1e-5, (
        "Total probability should be less than or close to one, but was "
        f"{total_probability}. Computed score probabilities: {probabilities}"
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


def extract_weighted_score(
    output: ModelOutput, min_score: int = 1, max_score: int = 10
) -> float:
    """
    Extract a weighted evaluation score from the model output.

    Args:
        output (ModelOutput): The model output containing logprobs.
        min_score (int): The minimum valid score. Defaults to 1.
        max_score (int): The maximum valid score. Defaults to 10.

    Returns:
        float: The weighted score.
    """
    target_score = extract_score(output.completion, min_score, max_score)

    def normalise_token(token: str) -> str:
        return token.strip().replace("▁", "").replace("_", "")

    def match_target_token(token: str) -> bool:
        return token == str(target_score)

    def token_parsing_function(token: str) -> int:
        score = int(token)
        if min_score <= score <= max_score:
            return score
        else:
            raise ValueError

    probabilities = _eval_weighted_options(
        output,
        normalise_token,
        match_target_token,
        token_parsing_function,
    )

    # Calculate the weighted score
    return sum([score * prob for score, prob in probabilities.items()])
