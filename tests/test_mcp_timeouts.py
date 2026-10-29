"""Exercise actual native transport configuration without model substitutes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm_coder import load_timed_mcp_toolsets


class McpTimeoutTests(unittest.TestCase):
    def load(self, directory, timeout):
        path = Path(directory) / "mcp.json"
        path.write_text(json.dumps({"mcpServers": {"body": {
            "url": "http://127.0.0.1:6767/mcp", "headers": {"X-Source": "native-timeout-contract"},
            "requestTimeoutMs": timeout}}}), encoding="utf-8")
        return load_timed_mcp_toolsets(path)

    def test_body_deadline_survives_native_loading(self):
        with TemporaryDirectory() as directory:
            tools = self.load(directory, 660000)
            client = tools[0].wrapped.client
            self.assertEqual(client._session_kwargs["read_timeout_seconds"].total_seconds(), 660)
            self.assertEqual(client.transport.headers["X-Source"], "native-timeout-contract")

    def test_invalid_deadlines_fail_before_connection(self):
        with TemporaryDirectory() as directory:
            for value in (0, -1, True, "660000", float("inf"), float("nan")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    self.load(directory, value)
