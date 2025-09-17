# Welcome to EvalSense

EvalSense is a tool for systematic evaluation of large language models (LLMs) on open-ended generation tasks, with a particular focus on bespoke, domain-specific evaluations. Some of its key features include:

- **Broad model support.** Out-of-the-box compatibility with a wide range of local and API-based model providers, including [Ollama](https://github.com/ollama/ollama), [Hugging Face](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), [OpenAI](https://platform.openai.com/docs/api-reference/introduction), [Anthropic](https://docs.claude.com/en/home) and [others](https://inspect.aisi.org.uk/providers.html).
- **Evaluation guidance.** An [interactive evaluation guide](https://nhsengland.github.io/evalsense/guide) and automated meta-evaluation tools assist in selecting the most appropriate evaluation methods for a specific use-case, including the use of perturbed data to assess method effectiveness.
- **Interactive UI.** A [web-based interface](https://nhsengland.github.io/evalsense/docs/#web-based-ui) enables rapid experimentation with different evaluation workflows without requiring any code.
- **Advanced evaluation methods.** EvalSense incorporates recent LLM-as-a-Judge and hybrid [evaluation approaches](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/), such as [G-Eval](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/#evalsense.evaluation.evaluators.GEvalScoreCalculator) and [QAGS](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/#evalsense.evaluation.evaluators.QagsConfig), while also supporting more traditional metrics like [BERTScore](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/#evalsense.evaluation.evaluators.BertScoreCalculator) and [ROUGE](https://nhsengland.github.io/evalsense/docs/api-reference/evaluation/evaluators/#evalsense.evaluation.evaluators.RougeScoreCalculator).
- **Efficient execution.** Intelligent experiment scheduling and resource management minimise computational overhead for local models. For remote APIs, EvalSense uses asynchrnous parallel calls to maximise throughput.
- **Modularity and extensibility.** Key components and evaluation methods can be used independently or replaced with user-defined implementations.
- **Comprehensive logging.** All key aspects of evaluation are recorded in machine-readable logs, including model parameters, prompts, model outputs, evaluation results, and other metadata.

## Quick Start

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

## Acknowledgements

We thank the [Inspect AI development team](https://github.com/UKGovernmentBEIS/inspect_ai/graphs/contributors) for their work on the [Inspect AI library](https://inspect.aisi.org.uk/), which serves as a basis for EvalSense.
