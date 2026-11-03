"""Pinned reference backends for numerical tests. Run this file once to cache them."""

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

import evaluate

# Full Hub commits and Evaluate's hash of each script plus its imported helpers.
# These are test fixtures, not a change to production metric-loading policy.
METRICS = {
    "bleu": (
        "4bc9d73a0ebdfcb960e09da5166c86591fc219b0",
        "9e0985c1200e367cce45605ce0ecb5ede079894e0f24f54613fca08eeb8aff76",
    ),
    "rouge": (
        "1bc20e7e9df922e5dfb597c31a4ea8467a89caff",
        "6e5315f72865c2eaa764c8361360bb938740b9c120a2cf3a7ad218aa0ce452ed",
    ),
}


def reference_metric(name: str) -> evaluate.EvaluationModule:
    revision, expected_hash = METRICS[name]
    backend = evaluate.load(name, revision=revision, keep_in_memory=True)
    if getattr(backend, "_hash", None) != expected_hash:
        raise RuntimeError(
            f"Unexpected {name} test implementation; fixture not loaded."
        )
    return backend


@contextmanager
def metric_backend(name: str) -> Iterator[evaluate.EvaluationModule]:
    """Use real metric arithmetic, replacing only network-dependent selection."""
    backend = reference_metric(name)

    def load(path: str, **kwargs) -> evaluate.EvaluationModule:
        if path not in (name, f"evaluate-metric/{name}"):
            raise AssertionError(f"Expected the {name} backend, got {path}.")
        return backend

    with patch("evaluate.load", side_effect=load):
        yield backend


if __name__ == "__main__":
    for metric_name in METRICS:
        reference_metric(metric_name)
        print(f"Prepared {metric_name} at {METRICS[metric_name][0]}")
