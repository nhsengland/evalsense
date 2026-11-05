import json
import unittest

from evalsense.workflow.analysers.metric_correlation_analyser import (
    MetricKey,
    _metric_names,
)


def metric_keys() -> list[MetricKey]:
    return [
        (
            json.dumps({"evaluator_name": "judge", "evaluator_fingerprint": version}),
            "quality",
            None,
        )
        for version in ("previous", "current")
    ]


class MetricLabelTests(unittest.TestCase):
    def test_retained_configurations_explain_ambiguous_labels(self):
        keys = metric_keys()
        names = {key: "judge" for key in keys}
        with self.assertRaises(ValueError) as context:
            _metric_names(keys, names, {})
        message = str(context.exception)
        self.assertIn("ambiguous", message)
        self.assertIn("earlier evaluator configurations", message)
        self.assertIn("Use distinct evaluator/scorer names", message)
        self.assertIn("analyse configurations separately", message)
        self.assertEqual(len(keys), 2)
        self.assertEqual(len(names), 2)

    def test_distinct_names_preserve_both_configurations(self):
        keys = metric_keys()
        labels = _metric_names(
            keys,
            {keys[0]: "previous judge", keys[1]: "current judge"},
            {},
        )
        self.assertEqual(
            labels, ["previous judge / quality", "current judge / quality"]
        )

    def test_shared_alias_does_not_hide_configuration_collision(self):
        keys = metric_keys()
        aliases = {"quality": "renamed quality"}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _metric_names(keys, {key: "judge" for key in keys}, aliases)
        self.assertEqual(aliases, {"quality": "renamed quality"})


if __name__ == "__main__":
    unittest.main()
