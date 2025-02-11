import typer

datasets_app = typer.Typer(
    no_args_is_help=True,
    help="Manage datasets for LLMScope.",
)


@datasets_app.command(no_args_is_help=True)
def get(name: str):
    """
    Download and prepare a dataset.
    """
    print(f"Downloading and preparing dataset {name}.")


if __name__ == "__main__":
    datasets_app()
