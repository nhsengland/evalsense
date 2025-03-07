import os
from pathlib import Path

from platformdirs import user_cache_dir

# Application metadata
APP_NAME = "llmscope"
APP_AUTHOR = "NHS"
USER_AGENT = "LLMScope/0.1.0"

# Datasets
DEFAULT_VERSION_NAME = "default"
DEFAULT_HASH_TYPE = "sha256"
INPUT_COLUMN = "llmscope_model_input"
OUTPUT_COLUMN = "llmscope_model_output"

if "OPENAI_API_KEY" in os.environ:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = None

if "LLMSCOPE_CACHE_DIR" in os.environ:
    CACHE_PATH = Path(os.environ["LLMSCOPE_CACHE_DIR"])
else:
    CACHE_PATH = Path(user_cache_dir(APP_NAME, APP_AUTHOR))
DATA_PATH = CACHE_PATH / "datasets"
MODELS_PATH = CACHE_PATH / "models"

DATASET_CONFIG_PATHS = [Path(__file__).parent / "dataset_config"]
if "DATASET_CONFIG_PATH" in os.environ:
    for directory in os.environ["DATASET_CONFIG_PATH"].split(os.pathsep):
        DATASET_CONFIG_PATHS.append(Path(directory))
