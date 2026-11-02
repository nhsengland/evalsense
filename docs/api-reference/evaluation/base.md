# :material-magnify: Evaluation

::: evalsense.evaluation


## Metric implementation provenance

The built-in BLEU, ROUGE and BERTScore evaluators pin their Evaluate scripts to
full Hub commits and check the loaded content hash, including helper scripts.
A different cached fallback raises an error before scoring. Pins are defined in
`evalsense.evaluation.metric_implementation` and should be updated together with
numerical regression checks.

`evaluator.implementation.provenance` exposes the script revision, content hash,
Python version and selected installed runtime dependency versions. The same
manifest is saved in each score's `metric_provenance` metadata and in the
`EvaluationRecord.evaluator_provenance_json` field. It changes evaluation cache
identity and log filenames without invalidating generated outputs. Ordinary
metric names and numerical definitions are unchanged.

Legacy records remain readable but do not satisfy a new implementation identity.
The standard summary, correlation and meta-evaluation analysers reject versions
that would otherwise share labels. Analyse one implementation at a time, for
example in separate projects; do not silently discard historical results.
Corpus BLEU also rejects scores without the matching provenance before
recomputation. Custom evaluators retain their existing identity unless they
explicitly provide an implementation.

These manifests are not a complete environment lock. Retain `uv.lock`, use
`uv sync --frozen`, and restart after dependency changes. Pretrained model weights,
tokenizer assets, custom aggregators, hardware and nondeterministic execution are
outside the script/version check. Preserve immutable model snapshots separately.
