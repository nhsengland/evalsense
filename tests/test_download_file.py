from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from requests import Response
from requests.structures import CaseInsensitiveDict
from urllib3.response import HTTPResponse

from evalsense.utils.files import download_file


def response(status, body, headers=None):
    reply = Response()
    reply.status_code = status
    reply.headers.update(headers or {})
    reply.raw = HTTPResponse(body=BytesIO(body), preload_content=False)
    return reply


class DownloadFileTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.target = Path(directory) / "dataset.json"
        self.part = self.target.with_suffix(".json.part")
        self.url = "https://example.org/dataset.json"
        self.head = self.enterContext(
            patch("evalsense.utils.files.get_remote_file_headers")
        )
        self.head.return_value = CaseInsensitiveDict(
            {"Accept-Ranges": "bytes", "Content-Length": "6"}
        )
        self.get = self.enterContext(patch("evalsense.utils.files.requests.get"))

    def download(self, **kwargs):
        download_file(
            self.url, self.target, show_progress=False, max_attempts=1, **kwargs
        )

    def test_full_response_replaces_partial_file(self):
        self.part.write_bytes(b"abc")
        self.get.return_value = response(200, b"abcdef")

        self.download()

        self.assertEqual(self.get.call_args.kwargs["headers"]["Range"], "bytes=3-")
        self.get.assert_called_once()
        self.assertEqual(self.target.read_bytes(), b"abcdef")
        self.assertFalse(self.part.exists())

    def test_partial_response_appends_to_existing_bytes(self):
        self.part.write_bytes(b"abc")
        self.get.return_value = response(206, b"def", {"Content-Range": "bytes 3-5/6"})

        self.download()

        self.assertEqual(self.target.read_bytes(), b"abcdef")

    def test_unsolicited_partial_response_leaves_target_untouched(self):
        self.target.write_bytes(b"old")
        self.get.return_value = response(206, b"abc", {"Content-Range": "bytes 0-2/6"})

        with self.assertRaisesRegex(RuntimeError, "206"):
            self.download(force_download=True)

        self.assertEqual(self.target.read_bytes(), b"old")
        self.assertFalse(self.part.exists())
