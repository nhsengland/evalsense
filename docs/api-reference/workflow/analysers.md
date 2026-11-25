# :material-magnify: Result Analysers

## Comparing experiment configurations

The standard summary has one row per stored generation configuration with aggregate metrics. Compare or join rows by `generation_id`. Two rows can share the same model name, dataset and other display labels. For example, the same model run at temperatures 0 and 1 has two generation IDs. The ID identifies a configuration and can recur across runs of that configuration.

`dataset_version` and `experiment_name` give human-readable context. To inspect a row's temperature in an existing `project`, use the ID from the table:

```python
import json

for record in project.records.evaluation:
    if record.generation_id == selected_id:
        settings = json.loads(record.model_record.generation_args_json)
        print(record.evaluator_name, settings.get("temperature"))
```

Evaluators that score the same generation appear side by side. A unique metric keeps its usual `score/metric` column name. When several recorded aggregates share that name, their column labels include the evaluator name and an identifier derived from their settings. A missing combination has a null cell. A repeated value for the same generation and aggregate raises an error even when both scores are equal. The analyser also checks for incompatible metric implementations before building the table.

Older projects use the identity fields present in their records. If two runs were already saved under the same record, the summary cannot recover the missing distinction. Rerun the summary from the saved evaluation logs to update a previously exported table.

::: evalsense.workflow.analysers
