"""Check the two score tables displayed in the Experiments notebook."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import polars as pl


class ExperimentsNotebookTests(unittest.TestCase):
    def test_score_tables_identify_configurations_with_the_same_model_name(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/Experiments.ipynb"
        cells = [
            "".join(cell["source"])
            for cell in json.loads(path.read_text())["cells"]
            if cell["cell_type"] == "code"
        ]
        displays = [source for source in cells if source.startswith("summary_results.")]
        self.assertEqual(len(displays), 2)

        metrics = [
            "Sentence BLEU/BLEU (corpus)",
            "ROUGE-1/mean",
            "ROUGE-2/mean",
            "ROUGE-L/mean",
            "BERTScore F1/mean",
            "Ternary Ref. QAGS (Llama 3.1) Accuracy/mean",
            "Judge Ref. QAGS (Llama 3.1)/mean",
        ] + [
            f"G-Eval (F. {prompt}, {model})/mean"
            for prompt in ("Brief", "Detail")
            for model in ("Gemma 3 27B", "Llama 3.1 8B", "Qwen 3 14B")
        ]
        summary = pl.DataFrame(
            {
                "model": ["same", "same"],
                "generation_id": ["config-a", "config-b"],
                **{metric: [0.25, 0.75] for metric in metrics},
            }
        )
        scope = {
            "pl": pl,
            "summary_results": summary,
            "bleu_evaluator": SimpleNamespace(name="BLEU"),
        }
        for cell in displays:
            with self.subTest(cell=cell):
                table = eval(compile(cell, str(path), "eval"), scope)
                displayed_metrics = set(table.columns).intersection(metrics)
                self.assertTrue(displayed_metrics)
                for metric in displayed_metrics:
                    self.assertCountEqual(
                        table.select("generation_id", metric).rows(),
                        [("config-a", 0.25), ("config-b", 0.75)],
                    )
