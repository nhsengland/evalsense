# :material-magnify: Evaluators

::: evalsense.evaluation.evaluators

## G-Eval rating responses

EvalSense requires one complete non-negative decimal integer in the configured
rating range. Outer whitespace and leading zeroes are allowed. Fractions,
decimals, signs, JSON or Markdown wrappers, and explanatory text are rejected.
For a 1–10 scale, `8` is valid; `0/10`, `-3`, `8.5` and
`Rating scale 1 to 10. Final rating 9` all raise `ValueError`.

The endpoints must be non-negative integers with `min_score <= max_score`.
Normalisation also requires `min_score < max_score`. For sampled ratings,
normalisation uses `(rating - min_score) / (max_score - min_score)`; on a 1–10
scale, 1 becomes 0, 10 becomes 1, and 8 becomes 7/9. With `normalise=False`, the
validated integer is returned unchanged.

For example, configure a sampled-rating evaluator as follows. The mock model
name is an offline placeholder; select the intended judge provider for a real run.

```python
from evalsense.evaluation.evaluators import get_g_eval_evaluator
from evalsense.generation import ModelConfig

evaluator = get_g_eval_evaluator(
    prompt_template=(
        "Rate the candidate from 1 to 10. Return only the integer, "
        "without a denominator, sign or explanation. Candidate: {prediction}"
    ),
    model_config=ModelConfig("mockllm/judge"),
    min_score=1,
    max_score=10,
    logprobs=False,
)
```

The complete response is checked before optional log-probability weighting.
Malformed responses cannot enter the weighted-score fallback. Direct scorer
calls raise `ValueError`; the pipeline records an evaluation error instead of
substituting zero or a maximum score. No automatic format-repair retry is added.

Valid responses with `logprobs=True` retain the existing weighting behaviour and
logged fallback when weights cannot be read. Parsing validation does not resolve
the complete-rating versus token-prefix probability problem tracked in
[issue #68](https://github.com/nhsengland/evalsense/issues/68). Use
`logprobs=False` to request the sampled integer rather than that weighted estimate.

### Existing scores

`get_g_eval_evaluator` includes the `evalsense.g-eval.integer.v1` format marker
in its evaluation-cache identity and successful score metadata. New evaluations
therefore do not reuse scores produced under the old parsing policy; matching
generated outputs remain reusable. Stored logs are not edited or revalidated.
Re-score affected evaluations and keep the historical logs separately when
comparing results. Code that constructs a custom `Evaluator` around a scorer
factory must update its explicit evaluator version when changing its policy.


## BERTScore IDF weighting

`BertScoreCalculator` and `get_bertscore_evaluator` score one prediction/reference
pair at a time. They support only `idf=False`, which is already the default.
Requests for automatic IDF or precomputed IDF dictionaries raise `ValueError`
before loading the metric. Changing a calculator's `idf` attribute is also
validated before scoring; weighting is never silently disabled.

BERTScore estimates inverse document frequency from a reference corpus. With one
reference, every token in it appears in every document, so its smoothed IDF is
zero. These weights cannot be normalised into a valid weighted score. Increasing
`batch_size` only changes embedding batches, not the one-reference corpus passed
by this evaluator.

The Hugging Face Evaluate BERTScore wrapper also treats a non-empty `idf`
dictionary as an instruction to fit weights from the supplied references rather
than using it as fixed token weights. Accepting a dictionary through this wrapper
would therefore retain the same problem. Empty dictionaries are rejected too;
they must not silently select unweighted scoring.

For corpus-weighted scoring, use the native `bert_score.BERTScorer` with
`idf=True` and an explicitly chosen `idf_sents` reference corpus, or the native
`bert_score.score` function with a precomputed dictionary. Keep the corpus and
weights fixed across comparisons, and use token IDs from the matching tokenizer.
A corpus-aware EvalSense integration is not implemented by this change.

The default unweighted scores, component names and hash metadata are unchanged.
Previously stored weighted results are not repaired or revalidated; retain their
provenance and do not compare them as though they were corpus-weighted BERTScore.
To rerun without IDF, construct a new evaluator with `idf=False` and a new name
so that old name-based evaluation caches are not reused.

References:

- [BERTScore authors' guidance on IDF and small reference sets](https://github.com/Tiiiger/bert_score#practical-tips)
- [Evaluate's BERTScore implementation](https://huggingface.co/spaces/evaluate-metric/bertscore/blob/b0f3cfd/bertscore.py)
