import hashlib
from pathlib import Path
import unicodedata
import regex
import requests
from requests.utils import CaseInsensitiveDict

from tenacity import Retrying, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

from llmscope.constants import USER_AGENT, DEFAULT_HASH_TYPE


def to_safe_filename(name: str) -> str:
    """Converts a string to a safe filename.

    Args:
        name (str): The string to convert.

    Returns:
        (str): The safe filename.
    """
    name = unicodedata.normalize("NFKD", name)
    name = regex.sub(r"[^\w\s-_]", "-", name)
    name = regex.sub(r"[-\s]+", "-", name)
    return name


def get_remote_file_headers(
    url: str, max_attempts: int = 2
) -> CaseInsensitiveDict[str]:
    """Gets the HTTP headers of a remote file.

    Args:
        url (str): The URL of the file.
        max_attempts (int, optional): The maximum number of attempts to get the headers.
            Defaults to 2.

    Returns:
        (CaseInsensitiveDict[str]): The headers of the file.

    Raises:
        RuntimeError: If the headers cannot be determined in the maximum
            number of attempts.
    """
    try:
        for attempt in Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=32),
            reraise=True,
        ):
            headers = {"User-Agent": USER_AGENT}
            with attempt as _, requests.head(url, headers=headers) as response:
                response.raise_for_status()
                return response.headers
    except Exception as e:
        raise RuntimeError(f"Failed to get headers for {url}: {e}")
    assert False, "Unreachable code"


def verify_file(
    file_path: str | Path,
    expected_size: int | None = None,
    expected_hash: str | None = None,
    hash_type: str = DEFAULT_HASH_TYPE,
    show_progress: bool = True,
    chunk_size: int = 10 * 1024**2,
) -> bool:
    """Verifies the integrity of a file against the provided metadata.

    Args:
        file_path (str | Path): The path to the file to verify.
        expected_size (int, optional): The expected size of the file in bytes
            (skips checking size if None).
        expected_hash (str, optional): The expected hash of the file (skips checking
            hash if None).
        hash_type (str, optional): The hash algorithm to use. Defaults to "sha256".
        show_progress (bool, optional): Whether to show verification progress.
        chunk_size (int, optional): The size of each verification chunk in bytes.
            Defaults to 10MB.

    Returns:
        (bool): True if the file matches the expected metadata, False otherwise.

    Raises:
        ValueError: If the hash type is unsupported.
    """
    file_path = Path(file_path)
    file_name = file_path.name
    if not file_path.exists():
        return False

    if expected_size is not None and file_path.stat().st_size != expected_size:
        return False

    if expected_hash is not None:
        try:
            hash_func = hashlib.new(hash_type)
        except ValueError:
            raise ValueError(f"Unsupported hash type: {hash_type}")

        with (
            file_path.open("rb") as f,
            tqdm(
                total=expected_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=not show_progress,
                desc=f"Verifying {file_name}",
                leave=False,
            ) as progress,
        ):
            while chunk := f.read(chunk_size):
                hash_func.update(chunk)
                progress.update(len(chunk))

        computed_hash = hash_func.hexdigest()

        if computed_hash.lower() != expected_hash.lower():
            return False

    return True


def download_file(
    url: str,
    target_path: str | Path,
    resume_download: bool = True,
    force_download: bool = False,
    show_progress: bool = True,
    max_attempts: int = 2,
    expected_hash: str | None = None,
    hash_type: str = DEFAULT_HASH_TYPE,
    chunk_size: int = 10 * 1024**2,
) -> None:
    """Downloads a file from a URL.

    Args:
        url (str): The URL of the file to download.
        target_path (str | Path): The path to save the downloaded file.
        resume_download (bool, optional): Whether to resume a partially
            downloaded file.
        force_download (bool, optional): Whether to force the download even
            if the file already exists.
        show_progress (bool, optional): Whether to show download progress.
        max_attempts (int, optional): The maximum number of download attempts.
            Defaults to 2.
        expected_hash (str, optional): The expected hash of the downloaded file.
        hash_type (str, optional): The hash algorithm to use. Defaults to "sha256".
        chunk_size (int, optional): The size of each download chunk in bytes.
            Defaults to 10MB.

    Raises:
        RuntimeError: If the download fails after the maximum number of attempts.
        ValueError: If max_attempts is invalid.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be 1 or greater")

    target_path = Path(target_path)
    target_name = target_path.name
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to determine the size of the downloaded file and resume support
    remote_file_headers = get_remote_file_headers(url, max_attempts=max_attempts)
    file_size = int(remote_file_headers.get("Content-Length", 0))
    supports_resume = "Accept-Ranges" in remote_file_headers
    content_encoding = remote_file_headers.get("Content-Encoding", "")
    compression_encodings = ["gzip", "deflate", "br", "zstd"]
    if any(encoding in content_encoding for encoding in compression_encodings):
        supports_resume = False

    # Return early if file already downloaded
    if not force_download and verify_file(
        target_path,
        expected_size=file_size,
        expected_hash=expected_hash,
        hash_type=hash_type,
    ):
        return

    # Check existing partial download
    temp_path = target_path.with_suffix(target_path.suffix + ".part")
    already_downloaded_size = (
        temp_path.stat().st_size
        if temp_path.exists() and resume_download and supports_resume
        else 0
    )
    if already_downloaded_size > file_size:
        # Partial download is corrupted — larger file size than expected
        already_downloaded_size = 0
    headers = (
        {"Range": f"bytes={already_downloaded_size}-"}
        if already_downloaded_size > 0
        else {}
    )
    headers["User-Agent"] = USER_AGENT

    # Try to download the file
    try:
        for attempt in Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=32),
            reraise=True,
        ):
            with (
                attempt as _,
                requests.get(url, stream=True, headers=headers, timeout=10) as response,
            ):
                response.raise_for_status()

                mode = "ab" if already_downloaded_size > 0 else "wb"
                with (
                    open(temp_path, mode) as file,
                    tqdm(
                        total=file_size,
                        initial=already_downloaded_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        disable=not show_progress,
                        desc=f"Downloading {target_name}",
                        leave=False,
                    ) as progress,
                ):
                    for chunk in response.iter_content(chunk_size):
                        if chunk:
                            file.write(chunk)
                            file.flush()
                            progress.update(len(chunk))

                if not verify_file(
                    temp_path,
                    expected_hash=expected_hash,
                    hash_type=hash_type,
                ):
                    # temp_path.unlink()
                    raise RuntimeError(
                        f"Downloaded file {target_path} could not be verified."
                    )

                # Success, move the temporary file to the target path
                temp_path.replace(target_path)

    except Exception as e:
        raise RuntimeError(
            f"Download from {url} failed after {max_attempts} attempts: {e}"
        )
