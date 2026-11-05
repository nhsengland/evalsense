# :material-magnify: Evaluators

::: evalsense.evaluation.evaluators


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
