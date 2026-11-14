# Changelog

All notable changes to this project will be documented in this file.

Instructions on how to update this Changelog are available in the `Updating the Changelog` section of the [`CONTRIBUTING.md`](./CONTRIBUTING.md). This project follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Breaking changes
- Built-in BLEU, ROUGE and BERTScore evaluations now have implementation-aware cache identities. Historical entries remain readable but are not reused. The standard analysers reject mixed or unrecorded implementations under the same labels; analyse legacy results separately.
- The BLEU evaluator now records unsmoothed sentence BLEU-4 instead of unigram precision. The default scorer is `Sentence BLEU`; the aggregate is `<evaluator name> (corpus)`.

  Re-score under a new evaluator name or in a fresh project; existing caches and notebook outputs are unchanged. Update result-column selectors. The legacy `BleuPrecisionScoreCalculator` remains available for direct unigram-precision callers. Corpus BLEU arithmetic is unchanged; repeated-epoch aggregation remains tracked separately in #69.
- BERTScore rejects inputs exceeding a declared tokenizer/model limit or whose length capability cannot be determined, rather than returning scores for truncated text. XLNet's explicit no-fixed-limit declaration is supported; finite tokenizer limits still apply.
- Custom solvers, preprocessing functions, and sample mappers now require explicit cache versions. Existing name-only cache entries remain readable but are not reused.
- Model configuration rejects inline credential fields, non-public headers, credential-bearing URLs and opaque objects. Configure provider authentication through environment variables or its credential chain. Use plain public configuration values.
- Meta-evaluation now requires an explicit complete set of at least two tier IDs. Results are separated by experiment/scorer identity, with exclusion counts and confidence intervals. Duplicate observations and mixed epochs are rejected.
- Per-sample BERTScore now rejects `idf=True` and IDF dictionaries. Only `idf=False` is supported by the current backend.

### Enhancements
- Include declared configuration and component versions in generation, preprocessing and evaluation cache identities, with separate evaluator log files.
- Deliberate two-tier meta-evaluations report pairwise ordering accuracy, concordant/discordant counts and Wilson intervals on the mean-correlation scale. Ties remain explicit exclusions. Three-or-more-tier designs retain sample-level bootstrap intervals.

### Bug fixes
- Record the original generation or retry error when Inspect raises. Untracked project logs can no longer mark the failed call as successful.
- Pin Evaluate metric scripts to full commits, check their loaded content hashes (including cached fallbacks), and record Python/runtime package versions in scores and project records. BERTScore reloads and corpus BLEU use the same pinned definition.
- Use all four modified n-gram precisions and the brevity penalty in BLEU sample-level analyses.
- Validate prediction and reference token lengths, including special tokens, before BERTScore embedding computation.
- Pair metric correlations by generation, source run, sample and epoch; reject duplicate observations, preserve scorer identities and report finite-pair counts.
- Check model and judge options before records or model calls, including after mutation. UI validation errors no longer echo input; unsafe legacy UI configurations are hidden.
- Prevent incomplete two-tier and undefined meta-evaluation correlations from silently inflating the reported mean. Report failed/missing logs and unestimable intervals.
- Prevent BERTScore from fitting IDF weights to a single reference or silently ignoring a supplied IDF dictionary.

### Documentation
- Describe metric provenance and migration. Keep historical logs, retain the lockfile and use frozen installation for the project environment. Recreate evaluators in a fresh process after changing packages. Model/tokenizer files, custom metric code, hardware and numerical nondeterminism still require separate controls; these pins do not guarantee bitwise-identical results.
- Document the BERTScore length policy and XLNet tokenizer constraints in the component docstrings.

  Existing cached results are not repaired or revalidated. Use a fresh project or a new evaluator name when applying this policy to previously scored data; with versioned components, also update their version. Preserve original logs as historical results. The pipeline records length-check failures instead of successful prefix-only scores.
- Add result-management guidance, notebook cache notes and fingerprint field documentation.
- Restore correlation analyser argument documentation and consolidate pairing, count and label guidance in the API docstrings.
- Document authentication setup and the configuration boundary. Existing files, logs and backups are not scrubbed: rotate any previously exposed credentials and handle those artifacts separately. Secrets embedded in arbitrary prompts, innocuous field values or custom provider logs remain the caller's responsibility.
- Distinguish intended perturbations from independently checked severity in meta-evaluation results and the example notebook. Document conditional uncertainty and sample-manifest requirements.

  Existing logs are unchanged. Update analyser calls to pass the planned `expected_tiers` and the correct tier direction. Review all exclusion counts and `ci_status`; old two-column tables and pooled averages are not directly comparable. The web UI requires the original tier IDs when loading meta-evaluation results. Independent severity review remains an external step; the analyser can require review flags but does not create or verify them.

### Miscellaneous
- None

## v0.1.6

### Miscellaneous
- Updated local model cleanup to be compatible with the new vLLM instance management logic in Inspect AI.

## v0.1.5

### Miscellaneous
-  Move web UI dependencies to a separate `webui` extra.

## v0.1.4

### Breaking changes
- The API and the internal logic of dataset managers have been updated. The `priority` argument to the constructor is no longer used. Instead, all classes inheriting from `DatasetManager` should define a class-level `priority` attribute. Additionally, the file-based dataset managers, such as the built-in `AciBenchDatasetManager`, now preprocess all splits when retrieving the dataset. This is more expensive, but prevents issues with loading different splits.
- EvalSense no longer overrides `HF_HUB_CACHE` and `HF_XET_CACHE` according to `EVALSENSE_STORAGE_DIR` and does not expose `MODELS_PATH` as a constant. Instead, the standard HuggingFace directory is used for storing models. If you wish to change this directory, you can do so by setting the appropriate [environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) directly.

### Enhancements
- EvalSense now provides a web-based graphical user interface for experimenting with evaluation workflows. This can be accessed by running `evalsense webui`.
- EvalSense now supports meta-evaluation of different evaluation methods using `MetaResultAnalyser`.
- It is now possible to construct dataset managers using the `DatasetManager.create(...)` class method, which automatically instantiates the dataset manager associated with the passed dataset name. This is facilitated by an internal registry of dataset managers.
- A new `HuggingFaceDatasetManager` enables loading arbitrary datasets from Hugging Face Hub.

### Bug fixes
- Fix `method_filter_fun` behaviour in `MetricCorrelationAnalyser`.

## v0.1.3

### Breaking changes

- Make local model dependencies optional during package installation. If you are using EvalSense with local models, you should install `evalsense[all]` or `evalsense[local]` instead of `evalsense`.

## v0.1.2

Technical release: no user-facing changes.

## v0.1.1

Technical release: no user-facing changes.

## v0.1.0

Initial release.
