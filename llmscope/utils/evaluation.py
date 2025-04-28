from collections import defaultdict
import regex

from inspect_ai.model import ModelOutput
import numpy as np


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


def extract_weighted_score(
    output: ModelOutput, score: int, min_score: int = 1, max_score: int = 10
) -> float:
    """
    Extract a weighted evaluation score from the model output.

    Args:
        output (ModelOutput): The model output containing logprobs.
        score (int): The score to extract.
        min_score (int): The minimum valid score. Defaults to 1.
        max_score (int): The maximum valid score. Defaults to 10.

    Returns:
        float: The weighted score.
    """
    if output.choices[0].logprobs is None:
        raise ValueError("Cannot computed weighted score, logprobs are not available.")

    def normalise_token(token: str) -> str:
        return token.strip().replace("▁", "").replace("_", "")

    # First, identify matching tokens
    logprobs = output.choices[0].logprobs.content
    score_logprobs = None
    for logprob in logprobs:
        token_text = normalise_token(logprob.token)
        if token_text == str(score):
            score_logprobs = logprob
            break
    if score_logprobs is None:
        raise ValueError(
            "Unable to identify correct token for computing weighted score."
        )
    if score_logprobs.top_logprobs is None:
        raise ValueError("Top logprobs are not available for computing weighted score.")

    # Then, calculate the probability for each candidate score
    probabilities: dict[int, float] = defaultdict(float)
    total_probability = 0.0
    for logprob in score_logprobs.top_logprobs:
        clean_token = normalise_token(logprob.token)

        # Ignore non-numeric tokens
        try:
            score = int(clean_token)
        except ValueError:
            continue

        # Ignore scores outside the valid range
        if score < min_score or score > max_score:
            continue

        score_probability = np.exp(logprob.logprob)
        probabilities[score] += score_probability
        total_probability += score_probability

    assert total_probability > 0, "Total probability should be greater than zero."
    # Allow for probability slightly exceeding 1 due to floating point errors
    assert total_probability <= 1 + 1e-5, (
        "Total probability should be less than or close to one, but was "
        f"{total_probability}. Computed score probabilities: {probabilities}"
    )

    # Finally, calculate the weighted score
    weighted_score = sum(
        [score * (prob / total_probability) for score, prob in probabilities.items()]
    )
    return weighted_score
