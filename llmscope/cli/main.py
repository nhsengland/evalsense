import typer
from typing_extensions import Annotated

from llmscope.cli.datasets import datasets_app

app = typer.Typer(
    no_args_is_help=True,
    help="LLMScope: A tool for evaluating LLM performance on healthcare tasks.",
)
app.add_typer(datasets_app, name="datasets")


@app.command(no_args_is_help=True)
def run(
    model: Annotated[str, typer.Option("--model", "-m")],
    dataset: Annotated[str, typer.Option("--dataset", "-d")],
):
    """
    Run a model on a dataset.
    """
    print(f"Running model {model} on dataset {dataset}.")


if __name__ == "__main__":
    app()
