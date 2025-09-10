from dataclasses import dataclass, field


@dataclass(frozen=True)
class AppState:
    dataset_name: str = ""
    dataset_splits: tuple[str] = field(default_factory=tuple)
    dataset_version: str = ""
    input_field_name: str = "input"
    target_field_name: str = "target"
    choices_field_name: str = "choices"
    id_field_name: str = "id"
    metadata_fields: tuple[str] = field(default_factory=tuple)
