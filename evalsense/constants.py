import os
from pathlib import Path

from platformdirs import user_cache_dir

# Application metadata
APP_NAME = "evalsense"
APP_AUTHOR = "NHS"
USER_AGENT = "EvalSense/0.1.0"

# Datasets
DEFAULT_VERSION_NAME = "default"
DEFAULT_HASH_TYPE = "sha256"

if "OPENAI_API_KEY" in os.environ:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = None

if "EVALSENSE_STORAGE_DIR" in os.environ:
    STORAGE_PATH = Path(os.environ["EVALSENSE_STORAGE_DIR"])
else:
    STORAGE_PATH = Path(user_cache_dir(APP_NAME, APP_AUTHOR))
DATA_PATH = STORAGE_PATH / "datasets"
MODELS_PATH = STORAGE_PATH / "models"
PROJECTS_PATH = STORAGE_PATH / "projects"
if "HF_HUB_CACHE" not in os.environ:
    os.environ["HF_HUB_CACHE"] = str(STORAGE_PATH / "huggingface")

DATASET_CONFIG_PATHS = [Path(__file__).parent / "dataset_config"]
if "DATASET_CONFIG_PATH" in os.environ:
    for directory in os.environ["DATASET_CONFIG_PATH"].split(os.pathsep):
        DATASET_CONFIG_PATHS.append(Path(directory))
