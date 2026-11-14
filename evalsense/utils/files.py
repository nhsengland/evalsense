from contextlib import contextmanager
import hashlib
from pathlib import Path
import unicodedata
import regex
import requests
from requests.utils import CaseInsensitiveDict

from tenacity import Retrying, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

from evalsense.constants import USER_AGENT, DEFAULT_HASH_TYPE


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
            headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
            with (
                attempt as _,
                requests.head(
                    url, headers=headers, allow_redirects=True, timeout=10
                ) as response,
            ):
                response.raise_for_status()
                return response.headers
    except Exception as e:
        raise RuntimeError(f"Failed to get headers for {url}: {e}")
    assert False, "Unreachable code (included as a hint for type checking)"


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


def _identity_encoding(headers: CaseInsensitiveDict[str]) -> bool:
    return headers.get("Content-Encoding", "").strip().lower() in ("", "identity")


def _content_length(headers: CaseInsensitiveDict[str]) -> int | None:
    # iter_content writes decoded bytes. An encoded length counts bytes on the wire.
    if not _identity_encoding(headers) or "Transfer-Encoding" in headers:
        return None
    value = headers.get("Content-Length")
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"Invalid Content-Length: {value!r}")
    return int(value)


def _full_download_size(
    response: requests.Response, file_size: int | None
) -> int | None:
    length = _content_length(response.headers)
    if file_size is not None and length is not None and file_size != length:
        raise ValueError(
            f"GET Content-Length {length} differs from HEAD length {file_size}"
        )
    return length if length is not None else file_size


def _range_total(
    headers: CaseInsensitiveDict[str], offset: int, file_size: int | None
) -> int | None:
    """Return the complete size if this range covers the entire requested suffix."""
    match = regex.fullmatch(
        r"bytes ([0-9]+)-([0-9]+)/([0-9]+|\*)",
        headers.get("Content-Range", ""),
        regex.I,
    )
    if match is None or not _identity_encoding(headers):
        return None
    try:
        start, end = int(match[1]), int(match[2])
        total = file_size if match[3] == "*" else int(match[3])
        length = _content_length(headers)
    except ValueError:
        return None
    if start != offset or start > end or end + 1 != total:
        return None
    if file_size is not None and total != file_size:
        return None
    if length is not None and length != end - start + 1:
        return None
    return total


@contextmanager
def _download_response(
    url: str, headers: dict[str, str], partial_path: Path | None, file_size: int | None
):
    """Open the requested suffix, falling back once to a full GET if needed."""
    offset = 0
    if partial_path is not None:
        try:
            offset = partial_path.stat().st_size
        except FileNotFoundError:
            pass
        if file_size is not None and offset > file_size:
            offset = 0
    for start in ((offset, 0) if offset else (0,)):
        request_headers = {**headers, "Range": f"bytes={start}-"} if start else headers
        with requests.get(
            url, stream=True, headers=request_headers, timeout=10
        ) as reply:
            if start and reply.status_code == 416:
                continue
            reply.raise_for_status()
            if reply.status_code == 200:
                yield reply, 0, _full_download_size(reply, file_size)
                return
            if start and reply.status_code == 206:
                total = _range_total(reply.headers, start, file_size)
                if total is not None:
                    yield reply, start, total
                    return
                continue
            raise RuntimeError(f"Unexpected download status {reply.status_code}")


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
    file_size = _content_length(remote_file_headers)
    supports_resume = (
        remote_file_headers.get("Accept-Ranges", "").strip().lower() == "bytes"
        and _identity_encoding(remote_file_headers)
    )

    if (
        not force_download
        and (file_size is not None or expected_hash is not None)
        and verify_file(
            target_path,
            expected_size=file_size,
            expected_hash=expected_hash,
            hash_type=hash_type,
        )
    ):
        return

    # Check existing partial download
    temp_path = target_path.with_suffix(target_path.suffix + ".part")
    partial_path = temp_path if resume_download and supports_resume else None
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}

    # Try to download the file
    try:
        for attempt in Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=32),
            reraise=True,
        ):
            with (
                attempt as _,
                _download_response(url, headers, partial_path, file_size) as (
                    response,
                    write_offset,
                    expected_size,
                ),
            ):
                if expected_size is None and expected_hash is None:
                    raise RuntimeError(
                        "Cannot verify download: no decoded byte count or expected hash"
                    )
                mode = "ab" if write_offset else "wb"
                with (
                    open(temp_path, mode) as file,
                    tqdm(
                        total=expected_size,
                        initial=write_offset,
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
                    expected_size=expected_size,
                    expected_hash=expected_hash,
                    hash_type=hash_type,
                ):
                    temp_path.unlink()
                    raise RuntimeError(
                        f"Downloaded file {target_path} could not be verified."
                    )

                # Success, move the temporary file to the target path
                temp_path.replace(target_path)

    except Exception as e:
        raise RuntimeError(
            f"Download from {url} failed after {max_attempts} attempts: {e}"
        )
