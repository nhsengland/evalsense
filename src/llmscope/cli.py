import typer
from typing_extensions import Annotated

app = typer.Typer(
    no_args_is_help=True,
    help="LLMScope: A tool for evaluating LLM performance on healthcare tasks.",
)


@app.command(no_args_is_help=True)
def get_dataset(name: Annotated[str, typer.Option("--dataset", "-d")]):
    """
    Download and prepare a dataset.
    """
    print(f"Downloading and preparing dataset {name}.")


@app.command(no_args_is_help=True)
def run_model(
    model: Annotated[str, typer.Option("--model", "-m")],
    dataset: Annotated[str, typer.Option("--dataset", "-d")],
):
    """
    Run a model on a dataset.
    """
    print(f"Running model {model} on dataset {dataset}.")


if __name__ == "__main__":
    app()
