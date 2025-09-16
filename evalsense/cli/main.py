from typing import Annotated

import typer

from evalsense.webui.app import launch_webui

app = typer.Typer(
    no_args_is_help=True,
    help="EvalSense: Tools for domain-specific LLM (meta-)evaluation.",
)


@app.command()
def webui(
    password: Annotated[
        str | None,
        typer.Option("--password", help="Set a custom password for the web UI."),
    ] = None,
    no_auth: Annotated[
        bool,
        typer.Option(
            "--no-auth",
            help="Disable authentication. Not recommended for public networks.",
        ),
    ] = False,
    share: Annotated[
        bool,
        typer.Option(
            "--share",
            help="If True, enables Gradio public sharing. This will make the app publicly accessible over the internet. Use with caution.",
        ),
    ] = False,
):
    """Launches the EvalSense Gradio web UI."""
    launch_webui(password=password, no_auth=no_auth, share=share)


@app.callback()
def callback():
    pass


if __name__ == "__main__":
    app()
