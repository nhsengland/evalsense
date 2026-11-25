"""Configuration identity survives reopening JSON and Inspect evaluation logs."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import polars as pl
from inspect_ai.log import EvalMetric, EvalResults, EvalScore, write_eval_log

from evalsense.evaluation import ResultRecord
from evalsense.generation import ModelConfig
from evalsense.workflow import Project
from evalsense.workflow.analysers import TabularResultAnalyser
from test_meta_evidence import log, record, sample


class TabularProjectRoundTripTests(unittest.TestCase):
    def test_temperature_sweep_keeps_scores_and_ids_when_reopened(self):
        analyser = TabularResultAnalyser[pl.DataFrame]()
        round_trips = []
        for suffix in ("json", "eval"):
            with (
                TemporaryDirectory() as tmp,
                patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp)),
            ):
                project = Project("sweep")
                project.evaluation_log_path.mkdir(parents=True)
                for temperature, value in ((0, 0.2), (1, 0.9)):
                    model = ModelConfig(
                        "model", generation_args={"temperature": temperature}
                    )
                    key = record(0, model_record=model.record)
                    path = project.evaluation_log_path / f"score-{temperature}.{suffix}"
                    score = EvalScore(
                        name="quality",
                        scorer="quality",
                        metrics={"mean": EvalMetric(name="mean", value=value)},
                    )
                    result = log(
                        [sample(1, 0, value)],
                        results=EvalResults(
                            total_samples=1, completed_samples=1, scores=[score]
                        ),
                    )
                    write_eval_log(result, location=str(path))
                    project.update_record(
                        key, ResultRecord(status="success", log_location=str(path))
                    )

                initial = analyser(project)
                self.assertCountEqual(initial["quality/mean"].to_list(), [0.2, 0.9])
                self.assertEqual(initial["generation_id"].drop_nulls().n_unique(), 2)

                # Older project files predate the configuration fingerprints.
                metadata = project.project_path / Project.METADATA_FILE
                saved = json.loads(metadata.read_text())
                for key, _ in saved["evaluation"]:
                    for field in (
                        "generator_fingerprint",
                        "task_fingerprint",
                        "evaluator_fingerprint",
                        "evaluator_provenance_json",
                    ):
                        del key[field]
                metadata.write_text(json.dumps(saved))

                reopened = analyser(Project("sweep"))
                columns = ["generation_id", "quality/mean"]
                rows = reopened.select(columns).sort("quality/mean").rows()
                self.assertEqual(
                    rows, initial.select(columns).sort("quality/mean").rows()
                )
                round_trips.append(rows)

        self.assertEqual(round_trips[0], round_trips[1])
