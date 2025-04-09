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
                default=str,
                sort_keys=True,
                ensure_ascii=True,
            ),
            generation_args_json=json.dumps(
                filtered_generation_args,
                default=str,
                sort_keys=True,
                ensure_ascii=True,
            ),
        )
