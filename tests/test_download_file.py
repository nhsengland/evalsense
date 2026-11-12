import hashlib
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

    def test_invalid_range_restarts_without_appending(self):
        for value in (None, "invalid", "bytes 2-5/6", "bytes 3-4/6", "bytes 3-5/7"):
            with self.subTest(value=value):
                self.part.write_bytes(b"abc")
                self.get.reset_mock()
                self.get.side_effect = [
                    response(206, b"bad", {"Content-Range": value} if value else {}),
                    response(200, b"abcdef"),
                ]

                self.download(force_download=True)

                self.assertEqual(self.target.read_bytes(), b"abcdef")
                first, second = self.get.call_args_list
                self.assertEqual(first.kwargs["headers"]["Range"], "bytes=3-")
                self.assertNotIn("Range", second.kwargs["headers"])

    def test_unsatisfiable_range_restarts_with_full_request(self):
        self.part.write_bytes(b"abcdef")
        self.get.side_effect = [response(416, b""), response(200, b"abcdef")]

        self.download()

        self.assertEqual(self.target.read_bytes(), b"abcdef")
        self.assertNotIn("Range", self.get.call_args.kwargs["headers"])

    def test_download_checks_known_size_before_replacing_target(self):
        self.target.write_bytes(b"old")
        for body in (b"abcde", b"abcdefg"):
            with self.subTest(body=body):
                self.get.return_value = response(200, body)

                with self.assertRaisesRegex(RuntimeError, "could not be verified"):
                    self.download(force_download=True)

                self.assertEqual(self.target.read_bytes(), b"old")

    def test_get_length_checks_download_when_head_omits_size(self):
        del self.head.return_value["Content-Length"]
        self.target.write_bytes(b"")
        self.get.return_value = response(200, b"abc", {"Content-Length": "6"})

        with self.assertRaisesRegex(RuntimeError, "could not be verified"):
            self.download()

        self.assertEqual(self.target.read_bytes(), b"")

    def test_conflicting_response_lengths_leave_target_untouched(self):
        self.target.write_bytes(b"old")
        self.get.return_value = response(200, b"abc", {"Content-Length": "3"})

        with self.assertRaisesRegex(RuntimeError, "Content-Length"):
            self.download(force_download=True)

        self.assertEqual(self.target.read_bytes(), b"old")

    def test_range_total_checks_download_when_head_omits_size(self):
        del self.head.return_value["Content-Length"]
        self.part.write_bytes(b"abc")
        self.get.return_value = response(206, b"de", {"Content-Range": "bytes 3-5/6"})

        with self.assertRaisesRegex(RuntimeError, "could not be verified"):
            self.download()

        self.assertFalse(self.target.exists())

    def test_zero_byte_length_is_checked(self):
        self.head.return_value["Content-Length"] = "0"
        self.get.return_value = response(200, b"x")

        with self.assertRaisesRegex(RuntimeError, "could not be verified"):
            self.download()

        self.assertFalse(self.target.exists())

    def test_missing_byte_count_requires_hash(self):
        self.head.return_value.clear()
        self.get.return_value = response(200, b"abcdef")
        self.target.write_bytes(b"old")

        with self.assertRaisesRegex(
            RuntimeError, "no decoded byte count or expected hash"
        ):
            self.download()

        self.assertEqual(self.target.read_bytes(), b"old")

    def test_hash_verifies_decoded_file_when_length_is_unknown(self):
        for headers in ({}, {"Content-Encoding": "gzip", "Content-Length": "26"}):
            with self.subTest(headers=headers):
                self.head.return_value.clear()
                self.head.return_value.update(headers)
                self.get.return_value = response(200, b"abcdef", headers)

                self.download(
                    force_download=True,
                    expected_hash=hashlib.sha256(b"abcdef").hexdigest(),
                )

                self.assertEqual(self.target.read_bytes(), b"abcdef")
