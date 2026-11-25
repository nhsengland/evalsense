# Contributing
We love contributions! We've compiled these docs to help you understand our contribution guidelines. If you still have questions, please [contact us](mailto:datascience@nhs.net), we'd be super happy to help.

## Contents of this file

- [Code of conduct](#code-of-conduct)
- [Folder structure](#folder-structure)
- [Commit hygiene](#commit-hygiene)
- [Updating Changelog](#updating-changelog)

## Code of Conduct
Please read [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) before contributing.

## Commit hygiene

Please see the [GDS guidelines](https://gds-way.digital.cabinet-office.gov.uk/standards/source-code/working-with-git.html#working-with-git), which describe general recommendations regarding commits and commit messages.

## Updating the Changelog

If you open a GitHub pull request on this repo, please update `CHANGELOG.md` to reflect your contribution.

Add your entry under `Unreleased` as:
- `Breaking changes`
- `Enhancements`
- `Bug fixes`
- `Documentation`
- `Miscellaneous`

Internal changes to the project that are not part of the public API do not need changelog entries, for example fixing the CI build server.

These sections follow [semantic versioning](https://semver.org/spec/v2.0.0.html), where:

- `Breaking changes` corresponds to a `major` (1.X.X) change.
- `Enhancements` corresponds to a `minor` (X.1.X) change.
- `Bug fixes` corresponds to a `patch` (X.X.1) change.

See the [`CHANGELOG.md`](./CHANGELOG.md) for an example for how this looks. However, note that the project is currently using major version zero, which means that the public API is not yet fully stable, and that the standard semantic versioning conventions do not apply.

## Correctness tests

Install the locked base dependencies, prepare the reference metric scripts, then
run the suite. No model weights, live model API, private dataset or GPU is needed.

```sh
uv sync --frozen --no-dev
uv run --no-sync python tests/metric_fixtures.py
uv run --no-sync python scripts/run_correctness_tests.py
```

Fixture preparation downloads the pinned BLEU and ROUGE scripts and verifies their
content hashes, including imported helpers. This is separate from the test run,
which disables Hugging Face network access. A missing or different metric fixture
fails the suite rather than skipping the numerical checks. Tests use synthetic
texts and temporary project directories; they do not modify existing experiments.

The runner discovers `test_*.py` files with `unittest`. It fails on assertion/import
errors, no discovered tests, or an entirely skipped run. Optional dependency skips
in additional suites remain visible. To run a subset, use `--pattern`, for example
`--pattern test_judge_numerics.py`.

Use small fixtures with hand-calculated expected values and a stated tolerance.
Test observable results, cache reuse and saved records rather than notebook source
text or implementation details. Mock model responses and failures; use the actual
numerical implementation where feasible. Do not regenerate expected values from
the function under test or assert an incorrect result merely to make CI pass.

The baseline covers corpus BLEU counts/length penalties, ROUGE overlap and LCS,
judge probability weighting and malformed responses, complete synthetic statistical
designs, and model/dataset cache separation. It does not establish prompt-injection
resistance or replace the bug-specific suites in #74–#83. In particular, partial
sample alignment, small-sample uncertainty, QAGS validation, BERTScore, expanded
cache identity, credentials and provenance remain covered by their respective
fixes. Keep those tests alongside the fixes; discovery picks them up when merged.

CI runs Python 3.12/3.13 with locked dependencies and a Python 3.12 job at the stated
minimum Evaluate, Datasets, Inspect, Polars, NumPy and SciPy versions. This is a
compatibility check for those dependencies, not every possible version combination.
