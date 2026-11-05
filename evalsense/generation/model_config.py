from dataclasses import dataclass, field
from functools import total_ordering
import json
from typing import Any

from pydantic import BaseModel

from inspect_ai.model import GenerateConfigArgs, Model

from evalsense.utils.credentials import ModelCredentialError, validate_public_config


@total_ordering
class ModelRecord(BaseModel, frozen=True):
    """A record identifying a model.

    Attributes:
        name (str): The name of the model.
        model_args_json (str): The model arguments as a JSON string.
        generation_args_json (str): The generation arguments as a JSON string.
    """

    # We need to use JSON strings here to keep the record hashable.
    name: str
    model_args_json: str = "{}"
    generation_args_json: str = "{}"

    def __eq__(self, other: object) -> bool:
        """Checks if this record is equal to another record.

        Args:
            other (object): The other record to compare with.

        Returns:
            bool: True if the records are equal, False otherwise.
        """
        if not isinstance(other, ModelRecord) or type(self) is not type(other):
            return NotImplemented
        return (
            self.name == other.name
            and self.model_args_json == other.model_args_json
            and self.generation_args_json == other.generation_args_json
        )

    def __lt__(self, other: object) -> bool:
        """Checks if this record is less than another record.

        Args:
            other (object): The other record to compare with.

        Returns:
            bool: True if this record is less than the other, False otherwise.
        """
        if not isinstance(other, ModelRecord) or type(self) is not type(other):
            return NotImplemented
        return (self.name, self.model_args_json, self.generation_args_json) < (
            other.name,
            other.model_args_json,
            other.generation_args_json,
        )

    def __hash__(self) -> int:
        """Returns a hash of the record.

        Returns:
            int: The hash of the record.
        """
        return hash((self.name, self.model_args_json, self.generation_args_json))


@dataclass
class ModelConfig:
    """Public configuration for a model used in an experiment.

    Credentials belong in provider-managed environment variables/credential chains,
    not model_args or generation_args. Common credential fields, non-public HTTP headers,
    credential-bearing URLs and opaque objects are rejected before persistence or
    pipeline use. Checks repeat because callers can mutate the argument dictionaries.
    Credentials are never hashed, silently dropped or replaced by redaction placeholders.
    Public protocol/version headers are limited to Accept, Content-Type, User-Agent,
    api-version and anthropic-version. Other headers require provider-managed setup.

    Authentication already held privately by an Inspect provider is not inspected.
    Its public model_args/config are checked, but this does not sanitise arbitrary
    provider logs, prompts or historical project files. Never put secrets in public
    names or other innocuous-looking fields. Revoke previously exposed credentials.
    """

    model: str | Model = field(repr=False)
    model_args: dict[str, Any] = field(default_factory=dict, repr=False)
    generation_args: GenerateConfigArgs = field(
        default_factory=GenerateConfigArgs, repr=False
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check serialisable settings without echoing rejected keys or values."""
        if type(self.model_args) is not dict or type(self.generation_args) is not dict:
            raise ModelCredentialError(
                "Model and generation arguments must be dictionaries of public options."
            )
        validate_public_config(self.name)
        validate_public_config(self.model_args)
        validate_public_config(self.generation_args)
        if isinstance(self.model, Model):
            validate_public_config(self.model.model_args)
            validate_public_config(dict(self.model.config))
            validate_public_config(getattr(self.model, "_explicit_base_url", None))
            validate_public_config(getattr(self.model.api, "base_url", None))

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        if isinstance(self.model, str):
            return self.model
        return self.model.name

    @property
    def record(self) -> ModelRecord:
        """Returns a record of the model configuration."""
        self.validate()
        # Remove arguments not directly affecting the used model or generation
        # procedure, so that we can match equivalent records.
        filtered_model_args = {
            k: v
            for k, v in self.model_args.items()
            if k not in {"device", "gpu_memory_utilization", "download_dir"}
        }
        filtered_generation_args = {
            k: v
            for k, v in self.generation_args.items()
            if k not in {"max_connections"}
        }

        return ModelRecord(
            name=self.name,
            model_args_json=json.dumps(
                filtered_model_args,
                sort_keys=True,
                ensure_ascii=True,
            ),
            generation_args_json=json.dumps(
                filtered_generation_args,
                sort_keys=True,
                ensure_ascii=True,
            ),
        )
