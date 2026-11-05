# :octicons-ai-model-16: Generation

::: evalsense.generation

## Authentication

Configure authentication through the provider's environment variables or credential
chain. Keep `ModelConfig` arguments public; do not paste credentials into the web UI.
See [Inspect's provider setup](https://inspect.aisi.org.uk/providers.html).

`ModelConfig` documents the supported public values and header allowlist. These
checks do not scrub existing files or detect secrets hidden in arbitrary text.
For previously exposed keys and artifacts, follow the migration note in the changelog.
