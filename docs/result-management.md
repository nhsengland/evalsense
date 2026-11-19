# Result management

## Cache identity

Display names are not sufficient cache keys. Generation records now also include
fingerprints of the solver configuration, preprocessing version, and sample field
mapping. Evaluation records include the evaluator parameters and judge model
configuration. Changing an evaluator does not invalidate reusable generations.

## Versioning custom code

Inspect-decorated built-in solvers and scorers supply construction parameters.
The built-in EvalSense evaluator factories declare their metric options explicitly.
For custom code, supply a non-empty `version` and change it when the implementation,
closure state, external files, or other undeclared inputs change. EvalSense does
not attempt to infer arbitrary Python behavior from source code or object reprs.

```python
steps = GenerationSteps("custom generation", custom_solver, version="2")
preprocessor = TaskPreprocessor("normalise", preprocess, version="3")
task = TaskConfig(
    dataset_manager=manager,
    generation_steps=steps,
    task_preprocessor=preprocessor,
    field_spec=record_to_sample,
    field_spec_version="1",
)
evaluator = Evaluator("custom score", custom_scorer, version="4")
```

A custom scorer can additionally declare `cache_parameters` for its configurable
settings. QAGS prompt implementations are custom code, so pass `version` to
`get_qags_evaluator`. Custom metric functions likewise require the containing
evaluator's version. Missing versions for opaque components raise
`CacheIdentityError` instead of silently reusing a name-only identity.

Versions remain the caller's responsibility for external dependencies, including
prompt files and mutable dataset/model revisions. Changes to runtime options
passed directly through `eval_kwargs`, `eval_retry_kwargs`, or `score_kwargs` are
not inferred from a component's construction parameters; update the appropriate
component version when using those options to change experimental conditions.
This change does not provide full environment or arbitrary-code fingerprinting.

## Package and execution settings

Fingerprints include the exact EvalSense and Inspect package versions. Updating
one of these packages creates new cache identities, even when the configuration
is otherwise unchanged. This is conservative: a version number alone does not
establish that two implementations produce equivalent results. Other dependencies
and external inputs still require the explicit versioning described above.

BERTScore's `verbose` setting does not affect its cache identity. When `idf=False`,
`nthreads` is also excluded because no IDF preparation is performed. Both options
are still passed to the calculator.

BERTScore's `device` and `batch_size` remain part of its identity. PyTorch does not
[guarantee bitwise-identical results across devices and batch shapes](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html).
This policy does not treat those execution settings as interchangeable.

## Preprocessed data

Preprocessed JSON files are keyed by the dataset record, Hugging Face dataset-state
fingerprint, and preprocessing identity, including the task-data transport policy.
Mapping from JSON records to Inspect samples happens afterwards; changes to that
mapping invalidate generation while the intermediate JSON remains reusable.

The dataset fingerprint comes from Hugging Face's
[documented cache mechanism](https://huggingface.co/docs/datasets/v3.4.1/about_cache#fingerprint).
Access is isolated in one helper. Missing, blank or non-string fingerprints raise
`CacheIdentityError` before reading or writing a task cache; there is no name-only
fallback. A change to the transport policy creates a new task-cache path even
when the dataset fingerprint is unchanged.

## Numerical task data

Task preprocessing writes JSONL from the stored Arrow values, preserving finite
floating-point values and integers without pandas decimal-precision rounding.
This applies to reference fields, nested metadata and records passed to a custom
`RecordToSample` mapper. Inspect still converts numeric targets to strings; their
numeric value is preserved. Precision already lost during dataset construction
or an explicit preprocessing step cannot be recovered by this transport.

Supported values are strings, booleans, integers, finite floats, `None`, lists or
tuples of supported values and dictionaries with string keys and supported values.
Lists and tuples are written as JSON arrays. Binary16 and binary32 values retain
their stored precision when promoted to Python floats. NaN,
infinities, decimals, timestamps, bytes and other unsupported types raise an
error identifying the row and field. They are never silently replaced with nulls
or rounded values. Convert them deliberately in the preprocessing function, for
example by choosing a string representation for a decimal or timestamp. The
`RecordToSample` mapper runs after serialisation, so such conversions must happen
before it.

The writer reads stored values independently of Hugging Face display formats and
lazy transforms, matching the previous export behaviour. Materialise intended
changes in the preprocessing function, for example with `Dataset.map`, and bump
its explicit `version` when the transformation changes. Use JSON-compatible
columns for custom feature types; a display formatter alone does not change the
stored records.

A task cache is replaced only after every record has been written successfully.
If writing fails, a new partial cache is removed and an existing cache remains
unchanged, including during `force_reprocess=True`.

### Worked example

These references remain nonzero through `FieldSpec`. The same preserved numbers
reach a custom mapper before it creates its Inspect samples.

```python
from math import isclose
from tempfile import TemporaryDirectory

from datasets import Dataset
from inspect_ai.dataset import FieldSpec

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.tasks import DefaultTaskPreprocessor

references = [1e-11, 2e-11, 4e-11]
source = Dataset.from_dict(
    {
        "prompt": ["Return the reference."] * len(references),
        "reference": references,
        "metadata": [{"reference": value} for value in references],
    }
)
with TemporaryDirectory() as directory:
    manager = HuggingFaceDatasetManager(
        "numeric-example", version="v1", splits=["test"], data_dir=directory
    )
    samples = DefaultTaskPreprocessor()(
        source, manager, FieldSpec(input="prompt", target="reference")
    )
    loaded = [float(sample.target) for sample in samples]
    assert loaded == references
    assert [sample.metadata["reference"] for sample in samples] == references

    def accuracy(predictions):
        return sum(
            isclose(prediction, target, rel_tol=1e-6, abs_tol=0.0)
            for prediction, target in zip(predictions, loaded, strict=True)
        ) / len(loaded)

    assert accuracy(references) == 1.0
    assert accuracy([0.0, 0.0, 0.0]) == 0.0
```

### Moving from rounded task caches

The `evalsense.task-jsonl.v1` transport policy is included in the preprocessor's
cache identity. Updated experiments use new task-data, generation and evaluation
identities, including experiments containing only text. This conservative change
prevents reuse of outputs and scores created from potentially rounded references.

Existing JSONL files, project records and logs remain available for inspection.
They are not repaired in place. Rerun affected experiments with the updated code
and original source data to produce new results; compare the records using their
full identities rather than their display names alone.

## Existing projects

Legacy project records and logs remain readable, but old name-only cache entries are not reused by newly configured experiments. Keep a
backup before using a project with older EvalSense releases, which do not know
about these identity fields.
