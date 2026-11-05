"""Synthetic credentials must not reach config displays, records or model calls."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, cast
import unittest
from unittest.mock import Mock, patch

from datasets import Dataset, DatasetDict
from inspect_ai.model import get_model
from inspect_ai.solver import generate

from evalsense.datasets.managers.huggingface import HuggingFaceDatasetManager
from evalsense.evaluation import Evaluator, ExperimentConfig, ResultRecord
from evalsense.generation import GenerationSteps, ModelConfig
from evalsense.utils.credentials import ModelCredentialError, validate_public_config
from evalsense.workflow import Pipeline, Project

MARKER = "synthetic-credential-46-not-a-real-secret"
PYTHON = {"max_model_len": 8192, "tokenizer_kwargs": {"eos_token": "</s>"}}


def experiment(config: ModelConfig, evaluator=None) -> ExperimentConfig:
    return ExperimentConfig(
        HuggingFaceDatasetManager("fixture", splits=["test"], version="1"),
        GenerationSteps("generate", generate()),
        config,
        evaluator=evaluator,
    )


class ModelConfigTests(unittest.TestCase):
    def assert_hidden(self, value):
        with self.assertRaises(ModelCredentialError) as caught:
            ModelConfig("mockllm/model", model_args=cast(Any, value))
        self.assertNotIn(MARKER, str(caught.exception))
        self.assertNotIn(MARKER, repr(caught.exception))
        self.assertIsNone(caught.exception.__cause__)

    def test_common_and_provider_specific_secrets_are_rejected(self):
        for key in (
            "api_key",
            "API-KEY",
            "apiKey",
            "openai_api_key",
            "password",
            "aws_access_key_id",
            "aws_secret_access_key",
            "azure_ad_token",
            "hf_token",
            "HuggingFaceToken",
            "client_secret",
            "private_key",
            "refresh_token",
            "Authorization",
            "Proxy-Authorization",
            "subscription_key",
            "credentials",
            "api%5Fkey",
        ):
            with self.subTest(key=key):
                self.assert_hidden({key: MARKER})

    def test_nested_credentials_and_header_containers_are_rejected(self):
        for value in (
            {"client_kwargs": {"api_key": MARKER}},
            {"providers": [{"Auth-Token": MARKER}]},
            {"extra_body": {"options": ({"clientSecret": MARKER},)}},
            {"headers": {"X-Custom-Provider-Field": MARKER}},
            {"default_headers": [("Authorization", MARKER)]},
            {"extra_headers": "Authorization: " + MARKER},
            {"cookies": {"session": MARKER}},
            {"auth": ("user", MARKER)},
        ):
            with self.subTest(shape=list(value)):
                self.assert_hidden(value)

    def test_url_credentials_and_authentication_values_are_rejected(self):
        for value in (
            f"https://user:{MARKER}@example.invalid/v1",
            f"//user:{MARKER}@example.invalid/v1",
            f"user:{MARKER}@example.invalid/v1",
            f"example.invalid/v1?api_key={MARKER}",
            f"https://example.invalid/v1?api_key={MARKER}",
            f"https://example.invalid/v1?access%5Ftoken={MARKER}",
            f"https://example.invalid/v1?X-Amz-Signature={MARKER}",
            f"https://example.invalid/v1#access_token={MARKER}",
            f"Bearer {MARKER}",
            f"Basic {MARKER}",
        ):
            with self.subTest(kind=value.split(":", 1)[0]):
                self.assert_hidden({"base_url": value})

    def test_ordinary_prompt_text_is_not_an_authentication_value(self):
        for text in (
            "Basic instructions for this evaluation.",
            "Digest the following paragraph before answering.",
            "See https://example.invalid/?key=public-example for instructions.",
        ):
            with self.subTest(text=text):
                config = ModelConfig(
                    "mockllm/model", generation_args=cast(Any, {"system_message": text})
                )
                self.assertEqual(
                    json.loads(config.record.generation_args_json)["system_message"],
                    text,
                )

    def test_generation_options_are_checked_too(self):
        for options in (
            {"extra_headers": {"Authorization": MARKER}},
            {"extra_body": {"api_key": MARKER}},
        ):
            with self.assertRaises(ModelCredentialError):
                ModelConfig("mockllm/model", generation_args=cast(Any, options))

    def test_public_headers_are_preserved_but_custom_headers_are_rejected(self):
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        config = ModelConfig("mockllm/model", model_args={"extra_headers": headers})
        self.assertEqual(
            json.loads(config.record.model_args_json)["extra_headers"], headers
        )
        self.assert_hidden({"request_headers": {"X-Custom": MARKER}})
        self.assert_hidden({"http_auth": ("user", MARKER)})

    def test_opaque_objects_are_never_stringified(self):
        class Opaque:
            def __str__(self):
                raise AssertionError("Must not stringify a client")

            __repr__ = __str__

        for value in ({"client": Opaque()}, {Opaque(): "value"}):
            with self.assertRaises(ModelCredentialError):
                ModelConfig("mockllm/model", model_args=cast(Any, value))

    def test_argument_containers_must_be_dictionaries(self):
        for field in ("model_args", "generation_args"):
            for value in (None, [MARKER], json.dumps({"api_key": MARKER})):
                with (
                    self.subTest(field=field),
                    self.assertRaises(ModelCredentialError) as caught,
                ):
                    ModelConfig("mockllm/model", **{field: cast(Any, value)})
                self.assertNotIn(MARKER, str(caught.exception))

    def test_recursive_values_fail_without_recursion_error(self):
        value: dict[str, Any] = {}
        value["self"] = value
        with self.assertRaises(ModelCredentialError):
            ModelConfig("mockllm/model", model_args=cast(Any, value))

    def test_safe_values_and_cache_identity_are_preserved(self):
        args = PYTHON | {
            "base_url": "https://example.invalid/v1?api-version=1",
            "device": "cpu",
            "revision": "version-one",
        }
        options = {
            "temperature": 0.7,
            "max_tokens": 40,
            "max_connections": 8,
            "logit_bias": {123: 0.2},
            "stop_seqs": ["end"],
        }
        before = deepcopy((args, options))
        config = ModelConfig("mockllm/model", args, cast(Any, options))
        record = config.record
        self.assertEqual(
            json.loads(record.model_args_json),
            {k: v for k, v in args.items() if k != "device"},
        )
        self.assertEqual(
            json.loads(record.generation_args_json),
            {
                "temperature": 0.7,
                "max_tokens": 40,
                "logit_bias": {"123": 0.2},
                "stop_seqs": ["end"],
            },
        )
        self.assertEqual(before, (args, options))
        self.assertEqual(
            config.record,
            ModelConfig(
                "mockllm/model", dict(reversed(list(args.items()))), cast(Any, options)
            ).record,
        )
        self.assertNotEqual(
            record,
            ModelConfig(
                "mockllm/model", args | {"revision": "version-two"}, cast(Any, options)
            ).record,
        )

    def test_none_credentials_and_empty_headers_have_no_secret(self):
        validate_public_config({"api_key": None, "headers": {}, "extra_headers": None})

    def test_post_construction_mutations_cannot_reach_records_or_repr(self):
        for field in ("model_args", "generation_args"):
            config = ModelConfig("mockllm/model")
            getattr(config, field)["nested"] = {"api_key": MARKER}
            self.assertNotIn(MARKER, repr(config))
            with self.assertRaises(ModelCredentialError):
                _ = config.record

    def test_model_names_with_url_credentials_are_rejected(self):
        with self.assertRaises(ModelCredentialError):
            ModelConfig(f"https://user:{MARKER}@example.invalid")

    def test_supplied_models_public_options_are_checked(self):
        for field in ("model_args", "config", "base_url"):
            model = get_model("mockllm/model")
            if field == "model_args":
                model.model_args["api_key"] = MARKER
            elif field == "config":
                if not hasattr(model.config, "extra_headers"):
                    continue  # This option is absent in the oldest supported Inspect.
                model.config.extra_headers = {"Authorization": MARKER}
            else:
                model.api.base_url = f"https://user:{MARKER}@example.invalid"
            with self.subTest(field=field), self.assertRaises(ModelCredentialError):
                ModelConfig(model)

    def test_private_provider_authentication_is_not_copied_into_record(self):
        model = get_model("mockllm/model", api_key=MARKER)
        config = ModelConfig(model)
        self.assertNotIn(MARKER, config.record.model_dump_json())
        self.assertNotIn(MARKER, repr(config))

    def test_pipeline_checks_mutated_generator_and_judge_configs_before_work(self):
        for judge in (False, True):
            config = ModelConfig("mockllm/model")
            case = (
                experiment(
                    ModelConfig("mockllm/model"),
                    Evaluator("judge", cast(Any, Mock()), model_config=config),
                )
                if judge
                else experiment(config)
            )
            config.model_args["api_key"] = MARKER
            with self.subTest(judge=judge), self.assertRaises(ModelCredentialError):
                Pipeline(case, Mock(spec=Project))

    def test_model_load_rechecks_mutation_even_for_active_model(self):
        config = ModelConfig("mockllm/model")
        pipeline = Pipeline(experiment(config), Mock(spec=Project))
        pipeline._active_model_config = config
        pipeline._active_model = get_model("mockllm/model")
        config.model_args["api_key"] = MARKER
        with patch("evalsense.workflow.pipeline.get_model") as load:
            with self.assertRaises(ModelCredentialError):
                pipeline._load_model(config)
            load.assert_not_called()

    def test_config_rejection_creates_no_project_records(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evalsense.workflow.project.PROJECTS_PATH", Path(tmp)),
        ):
            project = Project("safe")
            config = ModelConfig("mockllm/model")
            case = experiment(config)
            config.model_args["api_key"] = MARKER
            with self.assertRaises(ModelCredentialError):
                project.update_record(case.generation_record, ResultRecord())
            contents = (project.project_path / project.METADATA_FILE).read_text()
            self.assertNotIn(MARKER, contents)
            self.assertEqual(len(project.records.generation), 0)

    def test_provider_environment_auth_does_not_enter_project_or_inspect_logs(self):
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
                        {"input": ["synthetic input"], "target": ["answer"]}
                    )
                }
            ).save_to_disk(manager.main_data_path)
            config = ModelConfig("mockllm/model")
            case = ExperimentConfig(
                manager, GenerationSteps("generate", generate()), config
            )
            project = Project("auth")
            calls = []

            def provider(**kwargs):
                calls.append(kwargs)
                return get_model(**kwargs, api_key=os.environ["EVALSENSE_TEST_API_KEY"])

            with (
                patch.dict(os.environ, {"EVALSENSE_TEST_API_KEY": MARKER}),
                patch("evalsense.workflow.pipeline.get_model", side_effect=provider),
            ):
                Pipeline(case, project).generate(
                    show_progress=False,
                    eval_kwargs={"display": "none", "log_format": "json"},
                )
            self.assertEqual(len(calls), 1)
            self.assertNotIn(MARKER, repr(calls))
            self.assertTrue(project.get_logs(type="generation", status="success"))
            for path in project.project_path.rglob("*.json"):
                self.assertNotIn(MARKER, path.read_text())
            reopened = Project("auth")
            self.assertIn(case.generation_record, reopened.records.generation)


@unittest.skipUnless(importlib.util.find_spec("gradio"), "webui extra not installed")
class ModelConfigUITests(unittest.TestCase):
    def test_dictionary_parser_rejects_credentials_and_hides_bad_syntax(self):
        import gradio as gr
        from evalsense.webui.utils import dict_parser

        for text in (
            f"{{'api_key': '{MARKER}'}}",
            f"{{'api_key': '{MARKER}'",
            f"['{MARKER}']",
        ):
            with self.subTest(kind=text[:2]):
                with self.assertRaises((gr.Error, ModelCredentialError)) as caught:
                    dict_parser(text)
                self.assertNotIn(MARKER, str(caught.exception))
                self.assertNotIn(
                    MARKER, "".join(traceback.format_exception(caught.exception))
                )
        self.assertEqual(dict_parser("{'temperature':0.7}"), {"temperature": 0.7})
        self.assertEqual(dict_parser(""), {})

    def test_display_hides_legacy_secrets_without_modifying_state(self):
        from evalsense.webui.utils import configuration_for_display

        cases = [
            {"model_args": json.dumps({"api_key": MARKER})},
            {"model_name": "mockllm/model", "model_args": {"api_key": MARKER}},
            {"model_args": {"extra_headers": {"X-Custom-Auth": MARKER}}},
            {"generation_args": {"extra_body": {"access_token": MARKER}}},
        ]
        for data in cases:
            original = deepcopy(data)
            display = configuration_for_display(data)
            self.assertNotIn(MARKER, json.dumps(display))
            self.assertIn("Hidden", display["configuration"])
            self.assertEqual(data, original)
        safe = {"model_name": "mockllm/model", "model_args": PYTHON}
        self.assertEqual(configuration_for_display(safe), safe)

    def test_judge_factory_rejects_before_creating_evaluator(self):
        from evalsense.webui.configurators.evaluators.g_eval import GEvalConfigurator

        with patch(
            "evalsense.webui.configurators.evaluators.g_eval.get_g_eval_evaluator"
        ) as create:
            with self.assertRaises(ModelCredentialError):
                GEvalConfigurator().instantiate_evaluator(
                    model_name="mockllm/model", model_args={"api_key": MARKER}
                )
            create.assert_not_called()

    def test_model_callback_never_adds_rejected_configuration(self):
        import gradio as gr
        from evalsense.webui.components.models import models_tab
        from evalsense.webui.state import get_initial_state

        with gr.Blocks(analytics_enabled=False) as app:
            state = get_initial_state()
            models_tab(gr.State(state))
        callback = next(
            fn.fn for fn in app.fns.values() if fn.fn and fn.fn.__name__ == "add_model"
        )
        assert callback is not None
        with self.assertRaises(ModelCredentialError):
            callback("mockllm/model", repr({"api_key": MARKER}), "{}", state)
        self.assertEqual(state["model_configs"], [])
        callback(
            "mockllm/model", "{'max_model_len':8192}", "{'temperature':0.7}", state
        )
        self.assertEqual(len(state["model_configs"]), 1)

    def test_evaluator_callback_does_not_echo_parse_input_or_retain_secrets(self):
        import gradio as gr
        from evalsense.webui.components.evaluators import evaluators_tab
        from evalsense.webui.state import get_initial_state
        from evalsense.webui.utils import dict_parser

        rendered = []

        def capture_render(*args, **kwargs):
            def decorate(fn):
                rendered.append(fn)
                return fn

            return decorate

        state = get_initial_state()
        with (
            gr.Blocks(analytics_enabled=False) as app,
            patch(
                "evalsense.webui.components.evaluators.gr.render",
                side_effect=capture_render,
            ),
        ):
            configurator = Mock()
            configurator.input_widget.return_value = [
                {
                    "component": gr.Textbox(),
                    "input_name": "model_args",
                    "parser": dict_parser,
                }
            ]
            evaluators_tab(gr.State(state))
            with patch(
                "evalsense.webui.components.evaluators.EvaluatorConfigurator.create",
                return_value=configurator,
            ):
                rendered[0]("G-Eval")
        callback = next(
            fn.fn
            for fn in app.fns.values()
            if fn.fn and fn.fn.__name__ == "add_evaluator"
        )
        assert callback is not None
        for text in (repr({"api_key": MARKER}), "{'api_key': '" + MARKER):
            with self.assertRaises(gr.Error) as caught:
                callback(text, state)
            self.assertNotIn(MARKER, str(caught.exception))
            self.assertEqual(state["evaluator_configs"], [])

    def test_model_and_judge_renderers_do_not_emit_legacy_credentials(self):
        import gradio as gr
        from evalsense.webui.components.models import models_tab
        from evalsense.webui.components.evaluators import evaluators_tab
        from evalsense.webui.state import get_initial_state

        for module, builder, field, config in [
            (
                "models",
                models_tab,
                "model_configs",
                {
                    "model_name": "mockllm/model",
                    "model_args": {"api_key": MARKER},
                    "generation_args": {},
                },
            ),
            (
                "evaluators",
                evaluators_tab,
                "evaluator_configs",
                {
                    "evaluator_name": "G-Eval",
                    "evaluator_args": {"model_args": {"api_key": MARKER}},
                },
            ),
        ]:
            rendered = []

            def capture_render(*args, **kwargs):
                def decorate(fn):
                    rendered.append(fn)
                    return fn

                return decorate

            state = get_initial_state()
            state[field] = [config]
            with (
                gr.Blocks(analytics_enabled=False),
                patch(
                    f"evalsense.webui.components.{module}.gr.render",
                    side_effect=capture_render,
                ),
            ):
                builder(gr.State(state))
                with patch(f"evalsense.webui.components.{module}.gr.JSON") as display:
                    rendered[-1](state)
                self.assertNotIn(MARKER, repr(display.call_args))
                self.assertIn("Hidden", repr(display.call_args))


if __name__ == "__main__":
    unittest.main()
