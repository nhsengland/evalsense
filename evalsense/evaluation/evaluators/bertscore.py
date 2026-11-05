import gc
from numbers import Integral
from typing import Any, override

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

from evalsense.evaluation import Evaluator, ScoreCalculator


class BertScoreLengthError(ValueError):
    """Input length cannot be scored within the model's declared context."""


def _input_limit(tokenizer: Any, config: Any) -> int | None:
    """Return a finite bound, or None for XLNet's explicit no-fixed-limit case."""
    # Transformers uses a very large integer when no tokenizer limit is known.
    from transformers import XLNetConfig
    from transformers.tokenization_utils_base import LARGE_INTEGER

    limits = [
        int(value)
        for value in (
            getattr(tokenizer, "model_max_length", None),
            getattr(config, "max_position_embeddings", None),
            getattr(config, "n_positions", None),
        )
        if isinstance(value, Integral)
        and not isinstance(value, bool)
        and 0 < int(value) < LARGE_INTEGER
    ]
    if limits:
        return int(min(limits))
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    # XLNet explicitly declares -1 rather than an absent/unknown position limit.
    # A tokenizer sentinel alone must not disable checks for other architectures.
    if (
        isinstance(config, XLNetConfig)
        and config.max_position_embeddings == -1
        and isinstance(tokenizer_limit, Integral)
        and not isinstance(tokenizer_limit, bool)
        and int(tokenizer_limit) >= LARGE_INTEGER
    ):
        return None
    raise BertScoreLengthError(
        "Cannot determine a finite BERTScore input limit or a supported explicit "
        "no-fixed-limit declaration. Use a model with a declared context limit "
        "or a supported XLNet configuration; validate long-input suitability separately."
    )


def _load_length_context(
    model_type: str, use_fast_tokenizer: bool
) -> tuple[Any, int | None]:
    # Use BERTScore's loader, including its SciBERT model-name handling. Imports
    # stay lazy so importing EvalSense does not require the transformers extra.
    from bert_score.utils import get_tokenizer
    from transformers import AutoConfig

    tokenizer = get_tokenizer(model_type, use_fast=use_fast_tokenizer)
    config = AutoConfig.from_pretrained(tokenizer.name_or_path)
    return tokenizer, _input_limit(tokenizer, config)


def _encoded_length(tokenizer: Any, text: str) -> int:
    from transformers import GPT2Tokenizer, RobertaTokenizer

    # Match bert_score.utils.sent_encode's stripping, special tokens and prefix
    # space behavior, but disable truncation so the complete length is visible.
    text = text.strip()
    if not text:
        return len(tokenizer.build_inputs_with_special_tokens([]))
    kwargs: dict[str, Any] = {
        "add_special_tokens": True,
        "truncation": False,
        "verbose": False,
    }
    if isinstance(tokenizer, (GPT2Tokenizer, RobertaTokenizer)):
        kwargs["add_prefix_space"] = True
    return len(tokenizer.encode(text, **kwargs))


def _validate_pairwise_idf(idf: bool) -> None:
    """Do not estimate corpus weights from the single pair supplied to compute."""
    if idf is not False:
        raise ValueError(
            "BERTScore IDF weighting requires a reference corpus. "
            "This evaluator scores one pair at a time and supports only idf=False. "
            "Automatic IDF and precomputed IDF dictionaries are not supported "
            "by this backend; use a corpus-aware BERTScore scorer for IDF weighting."
        )


class BertScoreCalculator(ScoreCalculator):
    """Compute BERTScore without silently discarding tokens from either text.

    Prediction and reference lengths are checked separately, including special
    tokens, before embedding computation. Counting follows BERTScore's tokenizer,
    surrounding-whitespace and prefix-space handling with truncation disabled.
    The smallest finite tokenizer/configuration limit applies. Overlength inputs
    raise BertScoreLengthError with counts, never the input text or a partial score.

    XLNet explicitly declares no fixed positional limit. When its tokenizer also
    has no finite bound, the complete inputs are passed to BERTScore without an
    arbitrary cap. A finite tokenizer limit still applies. Missing declarations
    in other configurations remain errors; a large tokenizer sentinel alone is
    not evidence that the model supports arbitrary lengths.

    Use the default slow tokenizer for unbounded XLNet with BERTScore 0.3.13:
    that backend passes the large length sentinel to fast tokenizers, which can
    overflow. Tokenizer/backend errors propagate without producing a score.
    No fixed positional limit does not guarantee enough memory or reliable
    long-document scores; validate the chosen model on the intended task.

    Within-limit scores and hash metadata are unchanged. The validation context
    is loaded lazily, keeps only the current model/tokenizer setting, and is reset
    on cleanup. This calculator does not truncate, segment documents or average
    chunk scores. Increasing batch_size does not extend the context limit.

    This per-sample implementation rejects IDF weighting because corpus
    weights cannot be estimated from a single reference. Only idf=False
    is supported.
    """

    def __init__(
        self,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        lang: str = "en",
        num_layers: int | None = None,
        idf: bool = False,
    ):
        """
        Initializes the BERTScore calculator.

        Args:
            model_type (str, optional): The model type to use for computing BERTScore.
                Defaults to "microsoft/deberta-xlarge-mnli", the currently best-performing
                model according to BERTScore authors.
            lang (str, optional): The language of the text. Defaults to "en".
            num_layers (int, optional): The layer of representations to use.
            idf (bool, optional): Must be False. Estimating IDF from a single
                reference is invalid, and the Evaluate backend does not use supplied
                IDF dictionaries as fixed weights. Defaults to False.

        Raises:
            ValueError: If idf is not False. No metric or model is loaded in that case.
        """
        _validate_pairwise_idf(idf)
        self.bertscore_module = evaluate.load("bertscore")
        self.model_type = model_type
        self.lang = lang
        self.num_layers = num_layers
        self.idf = idf
        self._length_context: tuple[str, bool, Any, int | None] | None = None

    def _validate_input_lengths(
        self, prediction: str, reference: str, use_fast_tokenizer: bool
    ) -> None:
        context = self._length_context
        if context is None or context[:2] != (self.model_type, use_fast_tokenizer):
            tokenizer, limit = _load_length_context(self.model_type, use_fast_tokenizer)
            context = (self.model_type, use_fast_tokenizer, tokenizer, limit)
            self._length_context = context
        _, _, tokenizer, limit = context
        for source, text in (("prediction", prediction), ("reference", reference)):
            count = _encoded_length(tokenizer, text)
            if limit is not None and count > limit:
                # Do not echo clinical text into logs or error messages.
                raise BertScoreLengthError(
                    f"BERTScore {source} has {count} tokens including special tokens; "
                    f"the declared input limit is {limit}. No score was computed. "
                    "Use a suitable long-context model or an explicitly validated "
                    "segmentation strategy."
                )

    @override
    def calculate(
        self,
        *,
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
        verbose=False,
        device=None,
        batch_size=64,
        nthreads=1,
        rescale_with_baseline=False,
        baseline_path=None,
        use_fast_tokenizer=False,
        **kwargs,
    ) -> Score:
        """
        Calculates BERTScore for the supplied model prediction and reference input.

        Args:
            prediction (str): The text of the prediction from the model.
            input (str, optional): The text of the model input. Ignored for BERTScore.
            reference (str, optional): The text of the reference input to compare against.
            metadata (dict[str, Any], optional): Metadata for the evaluation. Ignored for BERTScore.
            verbose (bool): Whether to turn on verbose mode.
            device (str, optional): The device to use for computing the contextual embeddings.
            batch_size (int): The batch size to use for computing the contextual embeddings.
            nthreads (int): The number of threads to use for computing the contextual embeddings.
            rescale_with_baseline (bool): Whether to rescale the BERTScore with pre-computed baseline.
            baseline_path (str, optional): Customized baseline file.
            use_fast_tokenizer (bool): The `use_fast` parameter passed to HF tokenizer.

        Raises:
            BertScoreLengthError: If either input exceeds the declared context or
                its length capability is unknown. XLNet's explicit no-fixed-limit
                declaration is supported. No partial score is returned.

        Returns:
            Score: Inspect AI Score with the calculated evaluation results.
        """
        if reference is None:
            raise ValueError(
                "Reference is required for computing BERTScore, but was None."
            )

        # Validate public IDF configuration before loading tokenizer resources.
        _validate_pairwise_idf(self.idf)
        _validate_pairwise_idf(kwargs.get("idf", False))
        self._validate_input_lengths(prediction, reference, use_fast_tokenizer)
        predictions = [prediction]
        references = [reference]

        result = self.bertscore_module.compute(
            predictions=predictions,
            references=references,
            lang=self.lang,
            model_type=self.model_type,
            num_layers=self.num_layers,
            verbose=verbose,
            idf=False,
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
        input: str | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
        verbose=False,
        device=None,
        batch_size=64,
        nthreads=1,
        rescale_with_baseline=False,
        baseline_path=None,
        use_fast_tokenizer=False,
        **kwargs,
    ) -> Score:
        """
        Calculates BERTScore for the supplied model prediction and reference input.

        Args:
            prediction (str): The text of the prediction from the model.
            input (str | None): The text of the model input. Ignored for BERTScore.
            reference (str, optional): The text of the reference input to compare against.
            metadata (dict[str, Any] | None): Metadata for the evaluation. Ignored for BERTScore.
            verbose (bool): Whether to turn on verbose mode.
            device (str | None): The device to use for computing the contextual embeddings.
            batch_size (int): The batch size to use for computing the contextual embeddings.
            nthreads (int): The number of threads to use for computing the contextual embeddings.
            rescale_with_baseline (bool): Whether to rescale the BERTScore with pre-computed baseline.
            baseline_path (str | None): Customized baseline file.
            use_fast_tokenizer (bool): The `use_fast` parameter passed to HF tokenizer.

        Raises:
            BertScoreLengthError: If either input exceeds the declared context or
                its length capability is unknown. XLNet's explicit no-fixed-limit
                declaration is supported. No partial score is returned.

        Returns:
            Score: Inspect AI Score with the calculated evaluation results.
        """
        # Generic calculator callers may supply options through kwargs.
        _validate_pairwise_idf(kwargs.get("idf", False))
        return self.calculate(
            prediction=prediction,
            reference=reference,
            verbose=verbose,
            device=device,
            batch_size=batch_size,
            nthreads=nthreads,
            rescale_with_baseline=rescale_with_baseline,
            baseline_path=baseline_path,
            use_fast_tokenizer=use_fast_tokenizer,
        )


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
    idf: bool = False,
    device: str | None = None,
    batch_size: int = 64,
    nthreads: int = 1,
    rescale_with_baseline: bool = False,
    baseline_path: str | None = None,
    use_fast_tokenizer: bool = False,
    version: str | None = None,
) -> Evaluator:
    """
    Returns a BERTScore evaluator.

    Both texts are checked against any finite tokenizer/model limit, including
    special tokens. XLNet's explicit no-fixed-limit declaration is supported;
    unknown configurations still fail. See BertScoreCalculator for the length
    policy and backend constraints. No automatic truncation is performed.

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
        idf (bool, optional): Must be False because this evaluator scores one
            sample at a time. Automatic IDF and precomputed IDF dictionaries are
            unsupported. Defaults to False (no IDF weighting).
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

        version (str | None): Cache version for custom code or external resources.

    Returns:
        Evaluator: The BERTScore evaluator.

    Raises:
        ValueError: If idf is not False. Use a corpus-aware BERTScore scorer
            when IDF weighting is required.
    """
    cache_parameters = {
        "method": "bertscore",
        "metrics": metrics,
        "model_type": model_type,
        "lang": lang,
        "num_layers": num_layers,
        "idf": idf,
        "device": device,
        "batch_size": batch_size,
        "rescale_with_baseline": rescale_with_baseline,
        "baseline_path": baseline_path,
        "use_fast_tokenizer": use_fast_tokenizer,
    }
    # Progress logging never changes scores. IDF worker count is irrelevant when
    # IDF is disabled; device and batch size still identify numerical execution.
    if idf is not False:
        cache_parameters["nthreads"] = nthreads
    if metrics is None:
        metrics = [
            {"BERTScore Precision": [mean()]},
            {"BERTScore Recall": [mean()]},
            {"BERTScore F1": [mean()]},
        ]

    calculator = BertScoreCalculator(
        model_type=model_type,
        lang=lang,
        num_layers=num_layers,
        idf=idf,
    )

    async def init_bertscore() -> None:
        async with concurrency("init_bertscore", 1):
            if not hasattr(calculator, "bertscore_module"):
                setattr(
                    calculator,
                    "bertscore_module",
                    evaluate.load("bertscore"),
                )

    def cleanup_bertscore() -> None:
        import torch

        del calculator.bertscore_module
        calculator._length_context = None
        gc.collect()
        torch.cuda.empty_cache()

    @scorer(
        name=name,
        metrics=metrics,
    )
    def bertscore_scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            await init_bertscore()

            return await calculator.calculate_async(
                prediction=state.output.completion,
                reference=target.text,
                verbose=verbose,
                device=device,
                batch_size=batch_size,
                nthreads=nthreads,
                rescale_with_baseline=rescale_with_baseline,
                baseline_path=baseline_path,
                use_fast_tokenizer=use_fast_tokenizer,
            )

        return score

    return Evaluator(
        version=version,
        cache_parameters=cache_parameters,
        name=name,
        scorer=bertscore_scorer(),
        cleanup_fun=cleanup_bertscore,
    )
