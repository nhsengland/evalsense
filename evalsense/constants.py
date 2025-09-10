import os
from pathlib import Path

from platformdirs import user_cache_dir

# Application metadata
APP_NAME = "evalsense"
APP_AUTHOR = "NHS"
USER_AGENT = "EvalSense/0"

# Datasets
DEFAULT_VERSION_NAME = "default"
DEFAULT_HASH_TYPE = "sha256"

if "EVALSENSE_STORAGE_DIR" in os.environ:
    STORAGE_PATH = Path(os.environ["EVALSENSE_STORAGE_DIR"])
else:
    STORAGE_PATH = Path(user_cache_dir(APP_NAME, APP_AUTHOR))
DATA_PATH = STORAGE_PATH / "datasets"
PROJECTS_PATH = STORAGE_PATH / "projects"

DATASET_CONFIG_PATHS = [Path(__file__).parent / "dataset_config"]
if "DATASET_CONFIG_PATH" in os.environ:
    for directory in os.environ["DATASET_CONFIG_PATH"].split(os.pathsep):
        DATASET_CONFIG_PATHS.append(Path(directory))
