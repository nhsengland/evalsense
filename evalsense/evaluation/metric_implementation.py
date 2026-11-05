"""Pinned metric scripts and the Python environment used to evaluate them."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from typing import Any

import evaluate


# Full Hugging Face Space commits and Evaluate's content hashes, including imported
# helper scripts. Updating either requires checking the source and numerical tests.
_METRICS = {
    "bleu": (
        "4bc9d73a0ebdfcb960e09da5166c86591fc219b0",
        "9e0985c1200e367cce45605ce0ecb5ede079894e0f24f54613fca08eeb8aff76",
        (),
    ),
    "rouge": (
        "1bc20e7e9df922e5dfb597c31a4ea8467a89caff",
        "6e5315f72865c2eaa764c8361360bb938740b9c120a2cf3a7ad218aa0ce452ed",
        ("rouge-score", "nltk", "absl-py", "six"),
    ),
    "bertscore": (
        "4c18f1c7d175f111e097194635a63ac25346a819",
        "cf4907b18f8f741f202232c0f8009a3bd49ff98802c245abcb6ea51a37a8c05b",
        (
            "bert-score",
            "transformers",
            "torch",
            "tokenizers",
            "safetensors",
            "sentencepiece",
        ),
    ),
}
_COMMON_PACKAGES = (
    "evalsense",
    "evaluate",
    "datasets",
    "inspect-ai",
    "numpy",
    "pyarrow",
    "huggingface-hub",
    "packaging",
)


@dataclass(frozen=True)
class MetricImplementation:
    """Immutable provenance for one built-in metric implementation.

    Captures a full script revision, its expected Evaluate content hash, Python,
    and installed versions of the metric's runtime dependencies. Missing optional
    packages are recorded as null. Recreate evaluators in a fresh process after
    changing the environment. No environment variables or filesystem paths are recorded.

    This does not pin pretrained model weights, tokenizer assets, hardware or
    nondeterministic kernels. Use immutable model snapshots and retain the lockfile
    for experiments that require those guarantees.

    Attributes:
        name (str): Supported metric identifier.
        revision (str): Full Hugging Face Space commit.
        source_hash (str): Expected Evaluate hash of the script and helper code.
        python (str): Python interpreter version.
        packages (tuple): Sorted package names and installed versions, or None
            for packages absent from this environment.
    """

    name: str
    revision: str
    source_hash: str
    python: str
    packages: tuple[tuple[str, str | None], ...]

    @classmethod
    def resolve(cls, name: str) -> "MetricImplementation":
        """Capture the supported script definition and installed package versions."""
        revision, source_hash, extra_packages = _METRICS[name]
        packages = []
        for package in sorted(set(_COMMON_PACKAGES + extra_packages)):
            try:
                resolved = version(package)
            except PackageNotFoundError:
                resolved = None
            packages.append((package, resolved))
        return cls(
            name, revision, source_hash, platform.python_version(), tuple(packages)
        )

    @property
    def provenance(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible description suitable for saved scores."""
        return {
            "schema": "evalsense.metric-implementation.v1",
            "repository": f"evaluate-metric/{self.name}",
            "revision": self.revision,
            "source_hash": self.source_hash,
            "python": self.python,
            "packages": dict(self.packages),
        }

    @property
    def provenance_json(self) -> str:
        """Canonical, hashable representation for evaluation identity."""
        return json.dumps(self.provenance, sort_keys=True, separators=(",", ":"))

    def load(self) -> Any:
        """Load the pinned script and reject an incompatible cached fallback.

        Evaluate can fall back to the latest locally cached module even when a
        revision was requested. Its resolved content hash must therefore match
        the supported implementation before any scores are computed. This is a
        reproducibility check, not a sandbox for untrusted metric code.
        """
        module = evaluate.load(
            f"evaluate-metric/{self.name}", revision=self.revision, module_type="metric"
        )
        if getattr(module, "_hash", None) != self.source_hash:
            raise ValueError(
                f"Loaded {self.name} source does not match pinned revision {self.revision}. "
                "Do not reuse a different cached metric; restore access to the pinned source."
            )
        return module
