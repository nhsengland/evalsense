# EvalSense

A framework for systematic evaluation of large language models on open-ended, domain-specific tasks.

[![PyPI](https://img.shields.io/pypi/v/evalsense)](https://pypi.org/project/evalsense/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://github.com/nhsengland/evalsense/blob/main/pyproject.toml)
[![CI](https://github.com/nhsengland/evalsense/actions/workflows/evalsense.yml/badge.svg)](https://github.com/nhsengland/evalsense/actions/workflows/evalsense.yml)
[![MIT licence](https://img.shields.io/badge/Licence-MIT-green)](https://github.com/nhsengland/evalsense/blob/main/LICENCE)

> [!NOTE]
> This project is under active development. APIs may change between versions. Check the [changelog](./CHANGELOG.md) before upgrading.

## Project leadership

EvalSense is owned and managed by NHS England's Data Science and Applied AI team.

| Role | Person |
| --- | --- |
| **Scientific and Technical Lead** | [Sylvester Kaczmarek (@sylvesterkaczmarek)](https://github.com/sylvesterkaczmarek) |
| **Creator** | [Adam Dejl (@adamdejl)](https://github.com/adamdejl) |

Sylvester leads current scientific and technical development and maintenance, including work on evaluation correctness, reproducibility and reliability. Adam originally created EvalSense through an NHS England PhD Data Science internship.

## About

EvalSense supports the evaluation of open-ended text generation, with a particular focus on bespoke, domain-specific methods. Its key features include:

- **Broad model support.** Out-of-the-box compatibility with a wide range of local and API-based model providers, including [Ollama](https://github.com/ollama/ollama), [Hugging Face](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), [OpenAI](https://platform.openai.com/docs/api-reference/introduction), [Anthropic](https://docs.claude.com/en/home) and [others](https://inspect.aisi.org.uk/providers.html).
- **Evaluation guidance.** An [interactive evaluation guide](https://nhsengland.github.io/evalsense/guide) and automated meta-evaluation tools assist in selecting the most appropriate evaluation methods for a specific use-case, including the use of perturbed data to assess method effectiveness.
- **Interactive UI.** A [web-based interface](https://nhsengland.github.io/evalsense/docs/#web-based-ui) enables rapid experimentation with different evaluation workflows without requiring any code.
- **Advanced evaluation methods.** EvalSense incorporates recent LLM-as-a-Judge and hybrid [evaluation approaches](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/), such as [G-Eval](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/#evalsense.evaluation.evaluators.GEvalScoreCalculator) and [QAGS](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/#evalsense.evaluation.evaluators.QagsConfig), while also supporting more traditional metrics like [BERTScore](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/#evalsense.evaluation.evaluators.BertScoreCalculator) and [ROUGE](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/#evalsense.evaluation.evaluators.RougeScoreCalculator).
- **Efficient execution.** Intelligent experiment scheduling and resource management minimise computational overhead for local models. For remote APIs, EvalSense uses asynchronous parallel calls to maximise throughput.
- **Modularity and extensibility.** Key components and evaluation methods can be used independently or replaced with user-defined implementations.
- **Comprehensive logging.** All key aspects of evaluation are recorded in machine-readable logs, including model parameters, prompts, model outputs, evaluation results, and other metadata.

More information about EvalSense can be found on its [homepage](https://nhsengland.github.io/evalsense/) and in its [documentation](https://nhsengland.github.io/evalsense/docs/).

_**Note:** Only public or fake data are shared in this repository._

## Project structure

- The main code for the EvalSense Python package can be found under [`evalsense/`](https://github.com/nhsengland/evalsense/tree/main/evalsense).
- The accompanying documentation is available in the [`docs/`](https://github.com/nhsengland/evalsense/tree/main/docs) folder.
- Code for the interactive LLM evaluation guide is located under [`guide/`](https://github.com/nhsengland/evalsense/tree/main/guide).
- Jupyter notebooks with the evaluation experiments and examples are located under [`notebooks/`](https://github.com/nhsengland/evalsense/tree/main/notebooks).

## Getting Started

### Installation

You can install the project using [pip](https://pip.pypa.io/en/stable/) by running the following command:

```bash
pip install evalsense
```

This will install the latest released version of the package from [PyPI](https://pypi.org/project/evalsense/) *without any optional dependencies*.

Depending on your use-case, you may want to install *additional dependencies* from the following groups:

- `webui`: For using the interactive web UI.
- `jupyter`: For running experiments in Jupyter notebooks (only needed if you don't already have the necessary libraries installed).
- `transformers`: For using models and metrics requiring the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library.
- `vllm`: For using models and metrics requiring [vLLM](https://docs.vllm.ai/en/stable/).
- `interactive`: For using EvalSense with interactive UI features (currently includes `webui` and `jupyter`).
- `local`: For installing all local model dependencies (currently includes `transformers` and `vllm`).
- `all`: For installing all optional dependencies.

For example, if you want to install EvalSense with all optional dependencies, you can run:

```bash
pip install "evalsense[all]"
```

If you want to use EvalSense with the interactive features (`interactive`) and Hugging Face Transformers (`transformers`), you can run:

```bash
pip install "evalsense[interactive,transformers]"
```

and similarly for other combinations.

### Installation for Development

To install the project for local development, you can follow the steps below:

To clone the repo:

`git clone git@github.com:nhsengland/evalsense.git`

To set up the Python environment for the project:

- Install [uv](https://github.com/astral-sh/uv) if it's not installed already
- `uv sync --all-extras`
- `source .venv/bin/activate`
- `pre-commit install`

Note that the code is formatted with [ruff](https://github.com/astral-sh/ruff) and type-checked by [pyright](https://github.com/microsoft/pyright) in `standard` type checking mode. For the best development experience, we recommend enabling the corresponding extensions in your preferred code editor.

To set up the Node environment for the LLM evaluation guide (located under [`guide/`](https://github.com/nhsengland/evalsense/tree/main/guide)):

- Install [node](https://nodejs.org/en/download) if it's not installed already
- Change to the `guide/` directory (`cd guide`)
- `npm install`
- `npm run start` to run the development server

See also the separate [README.md](https://github.com/nhsengland/evalsense/tree/main/guide/README.md) for the guide.

### Programmatic Usage

For examples illustrating the usage of EvalSense, please check the notebooks under the `notebooks/` folder:

- The [Demo notebook](https://github.com/nhsengland/evalsense/blob/main/notebooks/Demo.ipynb) illustrates a basic application of EvalSense to the ACI-Bench dataset.
- The [Experiments notebook](https://github.com/nhsengland/evalsense/blob/main/notebooks/Experiments.ipynb) illustrates more thorough experiments on the same dataset, involving a larger number of evaluators and models.
- The [Meta-Evaluation notebook](https://github.com/nhsengland/evalsense/blob/main/notebooks/Meta-Evaluation.ipynb) focuses on meta-evaluation on synthetically perturbed data, where the goal is to identify the most reliable evaluation methods rather than the best-performing models.

### Web-Based UI

To use the interactive web-based UI implemented in EvalSense, simply run

```
evalsense webui
```

after installing the package and its dependencies. Note that you need to install EvalSense with the `webui` extra (`pip install "evalsense[webui]"`) or an extra that includes it before running this command.

## Contributing

Contributions are welcome. Please review [CONTRIBUTING.md](./CONTRIBUTING.md) before opening a pull request.

1. Fork the repository.
2. Create a branch (`git checkout -b feature/short-description`).
3. Make and test your changes.
4. Commit and push your branch.
5. Open a pull request.

## Licence

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENCE](./LICENCE) for more information._

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government Licence v3.0][ogl].

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

## Contact

EvalSense is currently maintained by [Sylvester Kaczmarek](https://github.com/sylvesterkaczmarek), its Scientific and Technical Lead. For bugs, feature requests and contributions, please [open an issue](https://github.com/nhsengland/evalsense/issues/new/choose) or [submit a pull request](https://github.com/nhsengland/evalsense/compare). Report security vulnerabilities through a [private vulnerability report](https://github.com/nhsengland/evalsense/security/advisories/new).

To find out more about the [NHS England Data Science](https://nhsengland.github.io/datascience/) visit our [project website](https://nhsengland.github.io/datascience/our_work/) or get in touch at [datascience@nhs.net](mailto:datascience@nhs.net).

## Acknowledgements

We thank the [Inspect AI development team](https://github.com/UKGovernmentBEIS/inspect_ai/graphs/contributors) for their work on the [Inspect AI library](https://inspect.aisi.org.uk/), which serves as a basis for EvalSense.
