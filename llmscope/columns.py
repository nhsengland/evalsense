from dataclasses import dataclass, field


@dataclass
class Column:
    """Class for configuring properties of a dataset column used by LLMScope.

    Attributes:
        column_name (str): The name of the column in the dataset.
        required_for_eval (bool): Whether the column is required for evaluation.
            Defaults to False.
        metadata (dict | None): Metadata associated with the column. Defaults to None.
    """

    column_name: str
    required_for_eval: bool = False
    metadata: dict | None = None


@dataclass
class ColumnConfig:
    """Configuration for dataset columns used by LLMScope.

    Attributes:
        inputs (Column): Configuration for the input column.
        outputs (Column): Configuration for the output column.
        references (Column): Configuration for the reference column.
        contexts (Column): Configuration for the context column.
        retrieval_contexts (Column): Configuration for the retrieval context column.
        additional_eval_columns (dict[str, Column]): Additional columns to be used
            for evaluation.
    """

    inputs: Column = field(
        default_factory=lambda: Column("llmscope_input_column", required_for_eval=False)
    )
    outputs: Column = field(
        default_factory=lambda: Column(
            "llmscope_output_column", required_for_eval=False
        )
    )
    references: Column = field(
        default_factory=lambda: Column(
            "llmscope_reference_column", required_for_eval=False
        )
    )
    contexts: Column = field(
        default_factory=lambda: Column(
            "llmscope_context_column", required_for_eval=False
        )
    )
    retrieval_contexts: Column = field(
        default_factory=lambda: Column(
            "llmscope_retrieval_context_column", required_for_eval=False
        )
    )
    additional_eval_columns: dict[str, Column] = field(default_factory=dict)

    @property
    def all_columns(self) -> dict[str, Column]:
        """Returns all columns in the configuration."""
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "references": self.references,
            "contexts": self.contexts,
            "retrieval_contexts": self.retrieval_contexts,
            **self.additional_eval_columns,
        }
