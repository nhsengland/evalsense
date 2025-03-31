from dataclasses import dataclass, field
import json
from typing import Any

from pydantic import BaseModel

from inspect_ai.model import GenerateConfigArgs, Model


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

    @property
    def model_args(self) -> dict[str, Any]:
        """Returns the model arguments as a dictionary."""
        return json.loads(self.model_args_json)

    @property
    def generation_args(self) -> dict[str, Any]:
        """Returns the generation arguments as a dictionary."""
        return json.loads(self.generation_args_json)


@dataclass
class ModelConfig:
    """Configuration for a model to be used in an experiment."""

    model: str | Model
    model_args: dict[str, Any] = field(default_factory=dict)
    generation_args: GenerateConfigArgs = field(default_factory=GenerateConfigArgs)

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        if isinstance(self.model, str):
            return self.model
        return self.model.name

    @property
    def record(self) -> ModelRecord:
        """Returns a record of the model configuration."""
        return ModelRecord(
            name=self.name,
            model_args_json=json.dumps(
                self.model_args,
                default=str,
                sort_keys=True,
                ensure_ascii=True,
            ),
            generation_args_json=json.dumps(
                dict(**self.generation_args),
                default=str,
                sort_keys=True,
                ensure_ascii=True,
            ),
        )
