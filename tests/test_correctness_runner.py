"""Prove the command used by CI cannot silently pass a non-executed suite."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_correctness_tests.py"


class CorrectnessRunnerTests(unittest.TestCase):
    def run_fixture(self, source: str | None) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            if source is not None:
                (Path(tmp) / "test_fixture.py").write_text(source)
            return subprocess.run(
                [sys.executable, str(RUNNER), "--start-directory", tmp],
                text=True,
                capture_output=True,
                timeout=30,
            )

    def test_passing_suite_returns_success(self):
        result = self.run_fixture(
            "import unittest\nclass Check(unittest.TestCase):\n def test_ok(self): self.assertEqual(1, 1)\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Ran 1 test", result.stderr)

    def test_failed_assertion_and_import_error_fail_the_command(self):
        for source in (
            "import unittest\nclass Check(unittest.TestCase):\n def test_bad(self): self.assertEqual(1, 2)\n",
            "raise RuntimeError('synthetic import error')\n",
        ):
            with self.subTest(source=source):
                result = self.run_fixture(source)
                self.assertEqual(result.returncode, 1, result.stderr)

    def test_empty_and_all_skipped_suites_fail_the_command(self):
        for source in (
            None,
            "import unittest\n@unittest.skip('fixture')\nclass Check(unittest.TestCase):\n def test_skipped(self): pass\n",
        ):
            with self.subTest(source=source):
                result = self.run_fixture(source)
                self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main()
