import os
from pathlib import Path

from platformdirs import user_cache_dir

APP_NAME = "llmscope"
APP_AUTHOR = "NHS"

if "LLMSCOPE_DATA_DIR" in os.environ:
    DATA_PATH = Path(os.environ["LLMSCOPE_DATA_DIR"])
else:
    DATA_PATH = Path(user_cache_dir(APP_NAME, APP_AUTHOR)) / "datasets"

USER_AGENT = "LLMScope/0.1.0"
