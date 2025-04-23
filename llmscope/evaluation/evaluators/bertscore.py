from typing import Callable

import evaluate
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    mean,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import concurrency

from llmscope.evaluation import Evaluator

_bertscore_fun: evaluate.EvaluationModule | None = None


async def _load_bertscore() -> evaluate.EvaluationModule:
    """
    Lazily loads the BERTScore evaluation module.

    Returns:
        evaluate.EvaluationModule: The loaded BERTScore evaluation module.
    """
    async with concurrency("load_bertscore", 1):
        global _bertscore_fun
        if _bertscore_fun is None:
            _bertscore_fun = evaluate.load("bertscore")

    return _bertscore_fun


def bertscore_base_factory(
    model_type="microsoft/deberta-xlarge-mnli",
    lang="en",
    num_layers=None,
    verbose=False,
    idf: bool | dict[str, float] = False,
    device=None,
    batch_size=64,
    nthreads=1,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False,
) -> Callable[[], Scorer]:
    """
    Base factory function to create a BERTScore scorer.

    Args:
        model_type (str): The model type to use for computing BERTScore.
            Defaults to "microsoft/deberta-xlarge-mnli", the currently best-performing
            model according to BERTScore authors.
        lang (str): The language of the text. Defaults to "en".
        num_layers (int | None): The layer of representations to use. The
            default is the number of layers tuned on WMT16 correlation data, which
            depends on the `model_type` used.
        verbose (bool): Whether to turn on verbose mode. Defaults to `False`
        idf (bool | dict): Use IDF weighting — can be a precomputed IDF dictionary.
            Defaults to `False` (no IDF weighting).
        device (str | None): The device to use for computing the contextual
            embeddings. If this argument is not set or `None`, the model will be
            loaded on `cuda:0` if available.
        nthreads (int): The number of threads to use for computing the
            contextual embeddings. Defaults to `1`.
        batch_size (int): The batch size to use for computing the
            contextual embeddings. Defaults to `64`.
        rescale_with_baseline (bool): Whether to rescale the BERTScore with
            pre-computed baseline. The default value is `False`.
        baseline_path (str | None): Customized baseline file.
        use_fast_tokenizer (bool): The `use_fast` parameter passed to HF
            tokenizer. Defaults to `False`.

    Returns:
        Callable[[], Scorer]: The BERTScore scorer factory function.
    """

    def bertscore_base() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            if not target.text:
                raise ValueError(
                    "Non-empty target is required for BERTScore evaluation."
                )

            bertscore_module = await _load_bertscore()
            predictions = [state.output.completion]
            references = [target.text]
            result = bertscore_module.compute(
                predictions=predictions,
                references=references,
                lang=lang,
                model_type=model_type,
                num_layers=num_layers,
                verbose=verbose,
                idf=idf,
                device=device,
                batch_size=batch_size,
                nthreads=nthreads,
                rescale_with_baseline=rescale_with_baseline,
                baseline_path=baseline_path,
                use_fast_tokenizer=use_fast_tokenizer,
            )
            return Score(
                value={
                    "precision": result["precision"],  # type: ignore
                    "recall": result["recall"],  # type: ignore
                    "f1": result["f1"],  # type: ignore
                },
                answer=state.output.completion,
                metadata={
                    "hashcode": result["hashcode"],  # type: ignore
                },
            )

        return score

    return bertscore_base


def bertscore_factory(
    *,
    model_type="microsoft/deberta-xlarge-mnli",
    lang="en",
    num_layers=None,
    verbose=False,
    idf: bool | dict[str, float] = False,
    device=None,
    batch_size=64,
    nthreads=1,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False,
) -> Callable[[], Scorer]:
    """
    Factory function to create a BERTScore scorer.

    Args:
        model_type (str): The model type to use for computing BERTScore.
            Defaults to "microsoft/deberta-xlarge-mnli", the currently best-performing
            model according to BERTScore authors.
        lang (str): The language of the text. Defaults to "en".
        num_layers (int | None): The layer of representations to use.
        verbose (bool): Whether to turn on verbose mode.
        idf (bool | dict): Use IDF weighting — can be a precomputed IDF dictionary.
        device (str | None): The device to use for computing the contextual embeddings.
        batch_size (int): The batch size to use for computing the contextual embeddings.
        nthreads (int): The number of threads to use for computing the contextual embeddings.
        rescale_with_baseline (bool): Whether to rescale the BERTScore with pre-computed baseline.
        baseline_path (str | None): Customized baseline file.
        use_fast_tokenizer (bool): The `use_fast` parameter passed to HF tokenizer.

    Returns:
        Scorer: The BERTScore scorer.
    """
    bertscore_base = bertscore_base_factory(
        model_type=model_type,
        lang=lang,
        num_layers=num_layers,
        verbose=verbose,
        idf=idf,
        device=device,
        batch_size=batch_size,
        nthreads=nthreads,
        rescale_with_baseline=rescale_with_baseline,
        baseline_path=baseline_path,
        use_fast_tokenizer=use_fast_tokenizer,
    )

    @scorer(
        name="BERTScore",
        metrics=[{"precision": [mean()]}, {"recall": [mean()]}, {"f1": [mean()]}],
    )
    def bertscore() -> Scorer:
        return bertscore_base()

    return bertscore


def get_bertscore_evaluator(
    *,
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    num_layers: int | None = None,
    verbose: bool = False,
    idf: bool | dict[str, float] = False,
    device: str | None = None,
    batch_size: int = 64,
    nthreads: int = 1,
    rescale_with_baseline: bool = False,
    baseline_path: str | None = None,
    use_fast_tokenizer: bool = False,
) -> Evaluator:
    """
    Get a BERTScore evaluator.

    Args:
        model_type (str, optional): The model type to use for computing BERTScore.
            Defaults to "microsoft/deberta-xlarge-mnli", the currently best-performing
            model according to BERTScore authors.
        lang (str, optional): The language of the text. Defaults to "en".
        num_layers (int | None, optional): The layer of representations to use. The
            default is the number of layers tuned on WMT16 correlation data, which
            depends on the `model_type` used.
        verbose (bool, optional): Whether to turn on verbose mode. Defaults to `False`
        idf (bool | dict, optional): Use IDF weighting — can be a precomputed IDF dictionary.
            Defaults to `False` (no IDF weighting).
        device (str | None, optional): The device to use for computing the contextual
            embeddings. If this argument is not set or `None`, the model will be
            loaded on `cuda:0` if available.
        nthreads (int, optional): The number of threads to use for computing the
            contextual embeddings. Defaults to `1`.
        batch_size (int, optional): The batch size to use for computing the
            contextual embeddings. Defaults to `64`.
        rescale_with_baseline (bool, optional): Whether to rescale the BERTScore with
            pre-computed baseline. The default value is `False`.
        baseline_path (str | None, optional): Customized baseline file.
        use_fast_tokenizer (bool, optional): The `use_fast` parameter passed to HF
            tokenizer. Defaults to `False`.

    Returns:
        Evaluator: The BERTScore evaluator.
    """
    return Evaluator(
        name="BERTScore",
        scorer=bertscore_factory(
            model_type=model_type,
            lang=lang,
            num_layers=num_layers,
            verbose=verbose,
            idf=idf,
            device=device,
            batch_size=batch_size,
            nthreads=nthreads,
            rescale_with_baseline=rescale_with_baseline,
            baseline_path=baseline_path,
            use_fast_tokenizer=use_fast_tokenizer,
        )(),
    )
