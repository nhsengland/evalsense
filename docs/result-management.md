# Result management

## Cache identity

Display names are not sufficient cache keys. Generation records now also include
fingerprints of the solver configuration, preprocessing version, and sample field
mapping. Evaluation records include the evaluator parameters and judge model
configuration. Changing an evaluator does not invalidate reusable generations.

## Generation sources

A configuration key can identify several attempts made with the same settings.
Each evaluation result also stores `source_generation`, a `GenerationSource`
containing the resolved generation-log path and a SHA256 digest of the complete
file. This identifies the saved JSON or `.eval` artifact, including its attempt
metadata. Two attempts that produce the same text still have distinct source
paths. Changing the file contents at an existing path also changes its source.

The source reference is saved with successful and failed evaluations and survives
reopening the project. Inspect it without requesting another evaluation.

```python
result = project.get_record(experiment.evaluation_record)
if result is not None and result.source_generation is not None:
    print(result.source_generation.log_location)
    print(result.source_generation.sha256)
```

Source validation streams the complete generation file through SHA256. It avoids
loading the whole file into memory, but reads it again for each cache validation
and before accepting a successful score. Large logs and multiple evaluators add
file-reading work. The digest does not establish whether the model output or
experimental design is correct.

## Rerunning experiments

Choose the operation that matches the intended work.

| Operation | Behaviour |
| --- | --- |
| `pipeline.run(force_rerun=True)` | Generate new outputs and evaluate that new generation. |
| `pipeline.evaluate(force_rerun=True)` | Rescore the current successful generation without running generation again. |
| `pipeline.evaluate()` | Reuse a successful evaluation only when its source still matches the current generation. Otherwise initialise from the current generation and score it. |

Generation and evaluation can also run separately. The later evaluation refreshes
its source even without a second force flag.

```python
pipeline.generate(force_rerun=True)
pipeline.evaluate()
```

When a successful replacement generation is recorded, dependent successful
evaluations with an older or missing source become `started`. They are excluded
from `project.get_logs(type="evaluation", status="success")` until rescored. Their
previous evaluation log and source remain accessible until the record is
replaced. The project keeps one current record per configuration and does not
provide an append-only history. Back up original results separately when they
must be retained.

Before recording scoring success, the pipeline checks the generation source
again. If it has been removed or changed, the evaluation is recorded as an error.
Call `pipeline.evaluate()` again to score the latest successful generation.
This check does not provide full cross-process transaction guarantees, tracked
in [#61](https://github.com/nhsengland/evalsense/issues/61), or address retention
of earlier generation artifacts after a failed rerun, tracked in
[#62](https://github.com/nhsengland/evalsense/issues/62).

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

Older evaluation metadata can lack `source_generation`. Loading or reading those
records leaves them available without repairing their reported scores. When
evaluation is requested, a missing or different source causes a fresh copy of
the current successful generation to be scored. Its saved source can then be
reused on subsequent calls. Recheck earlier reports explicitly before relying
on them as measurements of the current generation.

## Rechecking earlier meta-evaluation rankings

Earlier versions of the Meta-Evaluation notebook used the analyser default, which treated higher tier IDs as better quality. If you used those results to choose an evaluator, reopen the original project and recompute the correlations from its saved evaluation logs. This example uses the three tiers from the supplied notebook and assumes higher evaluator scores mean better quality.

```python
from evalsense.workflow import Project
from evalsense.workflow.analysers import MetaResultAnalyser

project = Project("ACI-Bench Perturbation")
results = MetaResultAnalyser()(
    project,
    expected_tiers=[0, 1, 2],
    lower_tier_is_better=True,
)
print(results.select("evaluator", "metric_name", "avg_correlation", "n_included"))
```

Use the original project name, storage location and planned tier IDs for your study. The analysis reads existing scores without running generation or scoring again. Before revising a comparison, check `n_excluded`, `failed_logs`, `unavailable_logs` and `ci_status` in the full result. Changes in which samples qualified can also affect the new ranking.
