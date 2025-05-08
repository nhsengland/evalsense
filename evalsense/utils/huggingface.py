from contextlib import contextmanager

from datasets import (
    disable_progress_bars,
    enable_progress_bars,
    are_progress_bars_disabled,
)


@contextmanager
def disable_dataset_progress_bars():
    """Context manager to disable progress bars for Hugging Face datasets."""
    progress_bars_enabled = not are_progress_bars_disabled()
    if progress_bars_enabled:
        disable_progress_bars()
    try:
        yield
    finally:
        if progress_bars_enabled:
            enable_progress_bars()
