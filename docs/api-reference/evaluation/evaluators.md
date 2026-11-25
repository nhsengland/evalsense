# :material-magnify: Evaluators

::: evalsense.evaluation.evaluators

## G-Eval rating responses

EvalSense requires one complete non-negative decimal integer in the configured
rating range. Outer whitespace is allowed. Sampled ratings also accept leading
zeroes; weighted ratings require canonical labels such as `8`, without zero
padding. Fractions, decimals, signs, JSON or Markdown wrappers, and explanatory
text are rejected.
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
Direct scorer calls raise `ValueError` for invalid responses; the pipeline records
an evaluation error. No automatic format-repair retry is added.

### Probability-weighted ratings

With `logprobs=True` (the default), EvalSense requests one output token and one
completion, overriding any larger limits in the model configuration. Each allowed
rating must appear as a complete canonical integer token in that token's returned
`top_logprobs`. The sampled token must match the complete response and its reported
probability. Token whitespace and one leading `▁` marker are supported; leading
zeroes, repeated identical tokens and invalid probability mass are rejected.

`top_logprobs` must be a positive integer at least as large as the number of ratings
on the scale. For 1–10, request at least 10 alternatives. This is a necessary
budget, not a guarantee: non-rating tokens may occupy the returned slots, and the
provider may return fewer alternatives or impose its own limit.

Missing ratings, missing log probabilities and multi-token responses fail clearly.
Where supported, increase `top_logprobs` to include omitted low-probability ratings.
Choose a smaller scale with suitable token labels, or explicitly set
`logprobs=False` to use sampled integer ratings. EvalSense never silently switches
from weighted scoring to a sampled rating. Providers that cannot produce this
one-token evidence cannot use the weighted mode.

For each allowed rating, EvalSense adds the probabilities of its valid reported
token spellings and normalises by their combined mass. The score is the conditional
mean over those reported valid rating tokens. Top-k truncation can omit other token
spellings, so this remains an approximation. It does not estimate probabilities of
arbitrary complete responses or multiply probabilities along multi-token labels.

For example, suppose a 1–10 scale has the following complete token evidence:

| Rating | Reported probability |
| --- | ---: |
| 1 through 8, each explicitly present | 0 |
| 9 | 0.30 |
| 10 | 0.70 |

The raw weighted score is `9 × 0.30 + 10 × 0.70 = 9.7`. Normalisation gives
`(9.7 - 1) / 9`, approximately `0.9667`. If the judge instead represents `10` as
`"1", "0"`, its first-token probability is not evidence for the complete rating
`10`. The missing complete token causes an error; EvalSense does not report the
incorrect prefix calculation `9 × 0.30 + 1 × 0.70 = 3.4`.

Successful scores retain `raw_score`, `output_text`, `prompt` and `score_format`,
plus the following estimator metadata in the saved Inspect log:

| Field | Meaning |
| --- | --- |
| `estimator` | `weighted_rating_tokens` or `sampled_rating`, according to the mode actually used. |
| `estimator_format` | `evalsense.g-eval.estimator.v1` for both scoring modes. |
| `weighting_policy` | `evalsense.g-eval.single-token.v1` for weighted scores. |
| `rating_probabilities` | Normalised probabilities keyed by rating strings, for weighted scores. |
| `rating_probability_mass` | The total reported valid-token probability mass before normalisation, for weighted scores. |

This evidence survives reopening JSON and `.eval` logs. Failed weighted attempts
remain evaluation errors and do not become successful point scores. Valid response
syntax and valid probability evidence do not establish the accuracy of the judge.

### Existing scores

`get_g_eval_evaluator` includes the `evalsense.g-eval.integer.v1` format marker
in its evaluation-cache identity and successful score metadata. Both scoring modes
also include the `evalsense.g-eval.estimator.v1` estimator-format marker, so earlier
scores without estimator evidence are re-scored. Weighted evaluators additionally
include the `evalsense.g-eval.single-token.v1` weighting policy in their cache
identity. Matching generated outputs remain reusable, and historical logs remain
readable.

Stored logs are not edited or revalidated. Re-score affected evaluations and keep
the historical logs separately when comparing results. Code that constructs a
custom `Evaluator` around a scorer factory must update its explicit evaluator
version when changing its policy.


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
