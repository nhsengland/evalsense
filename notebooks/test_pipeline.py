import os

os.environ["LLMSCOPE_CACHE_DIR"] = "/vol/bitbucket/ad5518/llmscope_cache"
os.environ["CUDA_HOME"] = "/vol/cuda/12.4.0/"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

from inspect_ai.dataset import FieldSpec
from inspect_ai.model import GenerateConfigArgs
from inspect_ai.solver import generate, prompt_template, system_message

from llmscope.constants import MODELS_PATH
from llmscope.datasets.managers import AciBenchDatasetManager
from llmscope.evaluation import ExperimentBatchConfig, TaskConfig
from llmscope.evaluation.evaluators import bleu_evaluator
from llmscope.generation import GenerationSteps, ModelConfig
from llmscope.tasks import DefaultTaskPreprocessor
from llmscope.workflow import Pipeline, Project


def main():
    aci_dataset_manager = AciBenchDatasetManager(splits=["train"])

    system_prompt_template = "You are an expert clinical assistant specialising in the creation of medically accurate summaries from a dialogue between the doctor and patient."
    user_prompt_template = """Your task is to generate a clinical note based on a conversation between a doctor and a patient. Use the following format for the clinical note:

    1. **CHIEF COMPLAINT**: [Brief description of the main reason for the visit]
    2. **HISTORY OF PRESENT ILLNESS**: [Summary of the patient's current health status and any changes since the last visit]
    3. **REVIEW OF SYSTEMS**: [List of symptoms reported by the patient]
    4. **PHYSICAL EXAMINATION**: [Findings from the physical examination]
    5. **RESULTS**: [Relevant test results]
    6. **ASSESSMENT AND PLAN**: [Doctor's assessment and plan for treatment or further testing]

    **Conversation:**
    {prompt}

    **Note:**
    """

    aci_generation = GenerationSteps(
        name="Structured",
        solver=[
            system_message(system_prompt_template),
            prompt_template(user_prompt_template),
            generate(),
        ],
    )

    aci_field_spec = FieldSpec(
        input="dialogue",
        target="note",
        id="id",
        metadata=[
            "dataset",
            "encounter_id",
            "doctor_name",
            "patient_gender",
            "patient_age",
            "patient_firstname",
            "patient_familyname",
            "cc",
            "2nd_complaints",
        ],
    )
    dialogue_task_preprocessor = DefaultTaskPreprocessor(name="Dialogue")

    llama_config = ModelConfig(
        "vllm/meta-llama/Llama-3.1-8B-Instruct",
        model_args={
            "download_dir": MODELS_PATH,
            "device": "1",
            "gpu_memory_utilization": 0.9,
            "max_model_len": 8192,
        },
        generation_args=GenerateConfigArgs(
            seed=42,
            temperature=0.7,
            top_p=0.95,
            max_connections=128,
        ),
    )

    aci_task_config = TaskConfig(
        dataset_manager=aci_dataset_manager,
        generation_steps=aci_generation,
        field_spec=aci_field_spec,
        task_preprocessor=dialogue_task_preprocessor,
    )

    experiment_config = ExperimentBatchConfig(
        tasks=[aci_task_config],
        model_configs=[llama_config],
        evaluators=[bleu_evaluator],
    )

    aci_project = Project(name="ACI-Bench Evaluation", reset_project=True)

    aci_pipeline = Pipeline(
        experiments=experiment_config,
        project=aci_project,
    )

    aci_pipeline.run()


if __name__ == "__main__":
    main()
