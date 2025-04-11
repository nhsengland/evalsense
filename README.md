# LLMScope: LLM Evaluation
<div align="center">

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)
[![license: MIT](https://img.shields.io/badge/License-MIT-brightgreen)](https://github.com/nhsengland/llmscope/blob/main/LICENCE)
[![LLMScope status](https://github.com/nhsengland/llmscope/actions/workflows/llmscope.yml/badge.svg)](https://github.com/nhsengland/llmscope/actions/workflows/llmscope.yml)
[![Guide status](https://github.com/nhsengland/llmscope/actions/workflows/guide.yml/badge.svg)](https://github.com/nhsengland/llmscope/actions/workflows/guide.yml)
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
This repository holds a Python package enabling systematic evaluation of large language models (LLMs) on open-ended generation tasks, with a particular focus on healthcare and summarisation. It also includes supplementary documentation and assets related to the NHS England project on LLM evaluation, such as the code for an interactive LLM evaluation guide (located in the `guide/` directory`). You can find more information about the project in the [original project proposal](https://nhsx.github.io/nhsx-internship-projects/genai-eval/).

_**Note:** Only public or fake data are shared in this repository._

## Project Stucture

- The main code for the LLMScope Python package can be found under [`llmscope/`](https://github.com/nhsengland/llmscope/tree/main/llmscope).
- The accompanying documentation is available in the [`docs/`](https://github.com/nhsengland/llmscope/tree/main/docs) folder.
- Code for the interactive LLM evaluation guide is located under [`guide/`](https://github.com/nhsengland/llmscope/tree/main/guide).
- Jupyter notebooks with the evaluation experiments and examples are located under [`notebooks/`](https://github.com/nhsengland/llmscope/tree/main/notebooks).

## Getting Started

### Installation for Development

To install the project for local development, you can follow the steps below:

To clone the repo:

`git clone git@github.com:nhsengland/llmscope.git`

To setup the Python environment for the project:

- Install [uv](https://github.com/astral-sh/uv) if it's not installed already
- `uv sync --all-extras`
- `source .venv/bin/activate`
- `pre-commit install`

To setup the Node environment for the LLM evaluation guide (located under [`guide/`](https://github.com/nhsengland/llmscope/tree/main/guide)):

- Install [node](https://nodejs.org/en/download) if it's not installed already
- `npm install` in the `guide/` directory
- `npm run start` to run the development server

## Usage
{DESCRIPTION OF CODE}

### Outputs
{LIST AND DESCRIPTION OF OUTPUTS}

{NOTES ON REPRODUCIBILITY OF RESULTS}

### Datasets
{DESCRIPTION AND LINKS TO DATASETS}

{LINK TO FAKE DATA TO SUPPORT INITAIL CODE RUNS}

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
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

### Contact

To find out more about the [NHS England Data Science](https://nhsengland.github.io/datascience/) visit our [project website](https://nhsengland.github.io/datascience/our_work/) or get in touch at [datascience@nhs.net](mailto:datascience@nhs.net).

<!-- ### Acknowledgements -->
