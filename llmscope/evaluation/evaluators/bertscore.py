import gc
from typing import override

import evaluate
from inspect_ai.scorer import (
    Metric,
    Score,
    Scorer,
    Target,
    mean,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import concurrency
import torch

from llmscope.evaluation import Evaluator, ScoreCalculator


class BertScoreCalculator(ScoreCalculator):
    """Calculator for computing BERTScores."""

    def __init__(self):
        self.bertscore_module = evaluate.load("bertscore")

    @override
    def calculate(
        self,
        *,
        prediction: str,
        reference: str | None = None,
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
        **kwargs: dict,
    ) -> Score:
        """
        Calculates BERTScore for the supplied model prediction and reference input.

        Args:
            prediction (str): The text of the prediction from the model.
            reference (str, optional): The text of the reference input to compare against.
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
            Score: Inspect AI Score with the calculated evaluation results.
        """
        if reference is None:
            raise ValueError(
                "Reference is required for computing BERTScore, but was None."
            )

        predictions = [prediction]
        references = [reference]

        result = self.bertscore_module.compute(
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
                "BERTScore Precision": result["precision"][0],  # type: ignore
                "BERTScore Recall": result["recall"][0],  # type: ignore
                "BERTScore F1": result["f1"][0],  # type: ignore
            },
            answer=prediction,
            metadata={
                "hashcode": result["hashcode"],  # type: ignore
            },
        )

    @override
    async def calculate_async(
        self,
        *,
        prediction: str,
        reference: str | None = None,
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
        **kwargs: dict,
    ) -> Score:
        """
        Calculates BERTScore for the supplied model prediction and reference input.

        Args:
            prediction (str): The text of the prediction from the model.
            reference (str, optional): The text of the reference input to compare against.
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
            Score: Inspect AI Score with the calculated evaluation results.
        """
        return self.calculate(
            prediction=prediction,
            reference=reference,
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


_bertscore_calculator: None | BertScoreCalculator = None


async def _init_bertscore() -> BertScoreCalculator:
    """
    Lazily initialises the BERTScore calculator.

    Returns:
        BertScoreCalculator: The initialised BERTScore calculator.
    """
    async with concurrency("load_bertscore", 1):
        global _bertscore_calculator
        if _bertscore_calculator is None:
            _bertscore_calculator = BertScoreCalculator()

    return _bertscore_calculator


def _cleanup_bertscore() -> None:
    """
    Cleans up the BERTScore calculator.
    """
    global _bertscore_calculator
    if _bertscore_calculator is not None:
        del _bertscore_calculator
        _bertscore_calculator = None
        gc.collect()
        torch.cuda.empty_cache()


def get_bertscore_evaluator(
    *,
    name: str = "BERTScore",
    metrics: list[Metric | dict[str, list[Metric]]]
    | dict[str, list[Metric]]
    | None = None,
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
    Returns a BERTScore evaluator.

    Args:
        name (str): The name of the evaluator and scorer. Defaults to "BERTScore".
        metrics (list[Metric | dict[str, list[Metric]]] | dict[str, list[Metric]] | None):
            The metrics to use for the evaluation. If `None`, the default metrics
            will be used (BERTScore Precision, Recall, and F1 with mean aggregation).
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
    if metrics is None:
        metrics = [
            {"BERTScore Precision": [mean()]},
            {"BERTScore Recall": [mean()]},
            {"BERTScore F1": [mean()]},
        ]

    @scorer(
        name=name,
        metrics=metrics,
    )
    def bertscore_scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            bertscore_calculator = await _init_bertscore()
            return await bertscore_calculator.calculate_async(
                prediction=state.output.completion,
                reference=target.text,
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

        return score

    return Evaluator(
        name=name,
        scorer=bertscore_scorer(),
        cleanup_fun=_cleanup_bertscore,
    )
