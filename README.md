> [!NOTE]
> This project is under development. The API may undergo major changes between versions, so we recommend checking the [CHANGELOG](https://github.com/nhsengland/evalsense/blob/main/CHANGELOG.md) for any breaking changes before upgrading.

# EvalSense: LLM Evaluation

<div align="center">

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)
[![PyPI package version](https://img.shields.io/pypi/v/evalsense)](https://pypi.org/project/evalsense/)
[![license: MIT](https://img.shields.io/badge/License-MIT-brightgreen)](https://github.com/nhsengland/evalsense/blob/main/LICENCE)
[![EvalSense status](https://github.com/nhsengland/evalsense/actions/workflows/evalsense.yml/badge.svg)](https://github.com/nhsengland/evalsense/actions/workflows/evalsense.yml)
[![Guide status](https://github.com/nhsengland/evalsense/actions/workflows/guide.yml/badge.svg)](https://github.com/nhsengland/evalsense/actions/workflows/guide.yml)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=fff)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/-React-61DAFB?logo=react&logoColor=white&style=flat)](https://react.dev/)

</div>
<div align="center">

[![Python v3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![ESLint](https://img.shields.io/badge/ESLint-3A33D1?logo=eslint)](https://eslint.org/)

</div>

## About

EvalSense is a framework for systematic evaluation of large language models (LLMs) on open-ended generation tasks, with a particular focus on bespoke, domain-specific evaluations. Some of its key features include:

- **Broad model support.** Out-of-the-box compatibility with a wide range of local and API-based model providers, including [Ollama](https://github.com/ollama/ollama), [Hugging Face](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), [OpenAI](https://platform.openai.com/docs/api-reference/introduction), [Anthropic](https://docs.claude.com/en/home) and [others](https://inspect.aisi.org.uk/providers.html).
- **Evaluation guidance.** An [interactive evaluation guide](https://nhsengland.github.io/evalsense/guide) and automated meta-evaluation tools assist in selecting the most appropriate evaluation methods for a specific use-case, including the use of perturbed data to assess method effectiveness.
- **Interactive UI.** A [web-based interface](https://nhsengland.github.io/evalsense/docs/#web-based-ui) enables rapid experimentation with different evaluation workflows without requiring any code.
- **Advanced evaluation methods.** EvalSense incorporates recent LLM-as-a-Judge and hybrid [evaluation approaches](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/), such as [G-Eval](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/#evalsense.evaluation.evaluators.GEvalScoreCalculator) and [QAGS](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/#evalsense.evaluation.evaluators.QagsConfig), while also supporting more traditional metrics like [BERTScore](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/#evalsense.evaluation.evaluators.BertScoreCalculator) and [ROUGE](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/#evalsense.evaluation.evaluators.RougeScoreCalculator).
- **Efficient execution.** Intelligent experiment scheduling and resource management minimise computational overhead for local models. For remote APIs, EvalSense uses asynchronous parallel calls to maximise throughput.
- **Modularity and extensibility.** Key components and evaluation methods can be used independently or replaced with user-defined implementations.
- **Comprehensive logging.** All key aspects of evaluation are recorded in machine-readable logs, including model parameters, prompts, model outputs, evaluation results, and other metadata.

More information about EvalSense can be found on its [homepage](https://nhsengland.github.io/evalsense/) and in its [documentation](https://nhsengland.github.io/evalsense/docs/).

_**Note:** Only public or fake data are shared in this repository._

## Project Stucture

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

This will install the latest released version of the package from [PyPI](https://pypi.org/project/evalsense/).

Depending on your use-case, you may want to install additional optional dependencies from the following groups:

- `interactive`: For running experiments interactively in Jupyter notebooks (only needed if you don't already have the necessary libraries installed).
- `transformers`: For using models and metrics requiring the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library.
- `vllm`: For using models and metrics requiring [vLLM](https://docs.vllm.ai/en/stable/).
- `local`: For installing all local model dependencies (currently includes `transformers` and `vllm`).
- `all`: For installing all optional dependencies.

For example, if you want to install EvalSense with all optional dependencies, you can run:

```bash
pip install "evalsense[all]"
```

If you want to use EvalSense with Jupyter notebooks (`interactive`) and Hugging Face Transformers (`transformers`), you can run:

```bash
pip install "evalsense[interactive,transformers]"
```

and similarly for other combinations.

### Installation for Development

To install the project for local development, you can follow the steps below:

To clone the repo:

`git clone git@github.com:nhsengland/evalsense.git`

To setup the Python environment for the project:

- Install [uv](https://github.com/astral-sh/uv) if it's not installed already
- `uv sync --all-extras`
- `source .venv/bin/activate`
- `pre-commit install`

Note that the code is formatted with [ruff](https://github.com/astral-sh/ruff) and type-checked by [pyright](https://github.com/microsoft/pyright) in `standard` type checking mode. For the best development experience, we recommend enabling the corresponding extensions in your preferred code editor.

To setup the Node environment for the LLM evaluation guide (located under [`guide/`](https://github.com/nhsengland/evalsense/tree/main/guide)):

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

after installing the package and its dependencies.

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b amazing-feature`)
3. Commit your Changes (`git commit -m 'Add some amazing feature'`)
4. Push to the Branch (`git push origin amazing-feature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

## License

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

## Contact

This project is currently maintained by [@adamdejl](https://github.com/adamdejl). If you have any questions, suggestions for new features or want to report a bug, please [open an issue](https://github.com/nhsengland/evalsense/issues/new/choose). For security concerns, please file a [private vulnerability report](https://github.com/nhsengland/evalsense/security/advisories/new).

To find out more about the [NHS England Data Science](https://nhsengland.github.io/datascience/) visit our [project website](https://nhsengland.github.io/datascience/our_work/) or get in touch at [datascience@nhs.net](mailto:datascience@nhs.net).

## Acknowledgements

We thank the [Inspect AI development team](https://github.com/UKGovernmentBEIS/inspect_ai/graphs/contributors) for their work on the [Inspect AI library](https://inspect.aisi.org.uk/), which serves as a basis for EvalSense.
