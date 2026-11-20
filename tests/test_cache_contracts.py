"""Baseline cache separation and reuse, using temporary synthetic projects."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict
from inspect_ai.model import GenerateConfigArgs
from inspect_ai.solver import generate

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import ExperimentConfig
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.workflow import Pipeline, Project


class CacheContractTests(unittest.TestCase):
    def test_mapping_order_does_not_change_model_identity(self):
        a = ModelConfig(
            "mockllm/model", model_args={"revision": "one", "max_model_len": 64}
        )
        b = ModelConfig(
            "mockllm/model", model_args={"max_model_len": 64, "revision": "one"}
        )
        self.assertEqual(a.record, b.record)
        self.assertEqual(hash(a.record), hash(b.record))

    def test_generation_settings_and_dataset_identity_invalidate_reuse(self):
        manager = HuggingFaceDatasetManager("synthetic", version="1", splits=["test"])
        steps = GenerationSteps("generate", generate())
        first = ExperimentConfig(manager, steps, ModelConfig("mockllm/model"))
        variants = [
            ExperimentConfig(manager, steps, ModelConfig("mockllm/other")),
            ExperimentConfig(
                manager,
                steps,
                ModelConfig("mockllm/model", model_args={"revision": "other"}),
            ),
            ExperimentConfig(
                manager,
                steps,
                ModelConfig(
                    "mockllm/model", generation_args=GenerateConfigArgs(temperature=1)
                ),
            ),
            ExperimentConfig(
                manager,
                steps,
                ModelConfig(
                    "mockllm/model", generation_args=GenerateConfigArgs(seed=7)
                ),
            ),
            ExperimentConfig(
                HuggingFaceDatasetManager("synthetic", version="2", splits=["test"]),
                steps,
                first.model_config,
            ),
            ExperimentConfig(
                HuggingFaceDatasetManager("synthetic", version="1", splits=["train"]),
                steps,
                first.model_config,
            ),
        ]
        self.assertEqual(
            len({first.generation_record, *(v.generation_record for v in variants)}), 7
        )

    def test_successful_generation_is_reused_after_reload_but_temperature_is_not(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp) / "projects"),
        ):
            manager = HuggingFaceDatasetManager(
                "synthetic", version="1", splits=["test"], data_dir=tmp
            )
            DatasetDict(
                {
                    "test": Dataset.from_dict(
                        {"id": [1], "input": ["synthetic note"], "target": ["answer"]}
                    )
                }
            ).save_to_disk(manager.main_data_path)

            def config(temperature: float) -> ExperimentConfig:
                return ExperimentConfig(
                    manager,
                    GenerationSteps("generate", generate()),
                    ModelConfig(
                        "mockllm/model",
                        generation_args=GenerateConfigArgs(temperature=temperature),
                    ),
                )

            first = config(0.0)
            Pipeline(first, Project("cache-contract")).generate(
                show_progress=False,
                eval_kwargs={"display": "none", "log_format": "json"},
            )
            reopened = Project("cache-contract")
            stored = reopened.get_record(first.generation_record)
            assert stored is not None and stored.log_location is not None
            self.assertEqual(stored.status, "success")
            original_bytes = Path(stored.log_location).read_bytes()
            cached = Pipeline(config(0.0), reopened)
            with patch.object(
                cached,
                "_generate_on_dataset",
                side_effect=AssertionError("cached generation was rerun"),
            ):
                cached.generate(show_progress=False)
            self.assertEqual(Path(stored.log_location).read_bytes(), original_bytes)
            # A new temperature must reach generation, not the successful-cache branch.
            changed = Pipeline(config(1.0), reopened)
            with patch.object(changed, "_generate_on_dataset") as generate_stage:
                changed.generate(show_progress=False)
                generate_stage.assert_called_once()
            self.assertEqual(reopened.get_record(first.generation_record), stored)


if __name__ == "__main__":
    unittest.main()
