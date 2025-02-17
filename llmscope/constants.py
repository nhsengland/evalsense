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

if "LLMSCOPE_DATA_DIR" in os.environ:
    DATA_PATH = Path(os.environ["LLMSCOPE_DATA_DIR"])
else:
    DATA_PATH = Path(user_cache_dir(APP_NAME, APP_AUTHOR)) / "datasets"

DATASET_CONFIG_PATHS = [Path(__file__).parent / "dataset_config"]
if "DATASET_CONFIG_PATH" in os.environ:
    for directory in os.environ["DATASET_CONFIG_PATH"].split(os.pathsep):
        DATASET_CONFIG_PATHS.append(Path(directory))
