"""Discover regression tests; fail for failures, import errors or an empty run."""

import argparse
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-directory", type=Path, default=ROOT / "tests")
    parser.add_argument("--pattern", default="test_*.py")
    args = parser.parse_args()
    # Metric fixtures are prepared separately. Scoring must not fetch a different
    # implementation or download models during an ostensibly deterministic test.
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_EVALUATE_OFFLINE"):
        os.environ[name] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(ROOT))
    suite = unittest.TestLoader().discover(
        str(args.start_directory), pattern=args.pattern
    )
    if not suite.countTestCases():
        print("No correctness tests discovered.", file=sys.stderr)
        return 2
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun == len(result.skipped):
        print("No correctness tests executed; every test was skipped.", file=sys.stderr)
        return 2
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
