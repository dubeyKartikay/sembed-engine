#!/usr/bin/env python3

import argparse
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


def write_dataset(path, record_ids, vectors):
    dimensions = len(vectors[0])
    with path.open("wb") as output:
        output.write(struct.pack("=qq", len(vectors), dimensions + 1))
        output.write(struct.pack(f"={len(record_ids)}q", *record_ids))
        for vector in vectors:
            output.write(struct.pack(f"={dimensions}f", *vector))


class SembedIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = SEMBED_BINARY

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.dataset = self.directory / "vectors.bin"
        self.index = self.directory / "vectors.sembed"
        write_dataset(
            self.dataset,
            [101, 102, 103, 104, 105, 106],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0],
             [6.0, 5.0], [5.0, 6.0]],
        )

    def tearDown(self):
        self.temporary.cleanup()

    def run_sembed(self, *arguments, input_text=None, check=True):
        return subprocess.run(
            [str(self.binary), *map(str, arguments)],
            input=input_text,
            capture_output=True,
            text=True,
            check=check,
        )

    def build_index(self):
        completed = self.run_sembed(
            "index",
            "--dataset",
            self.dataset,
            "--output",
            self.index,
            "--degree-threshold",
            3,
            "--search-list-size",
            6,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["records"], 6)
        self.assertEqual(result["dimensions"], 2)

    def test_saved_index_is_self_contained_and_accepts_full_vector(self):
        self.build_index()
        self.dataset.unlink()

        completed = self.run_sembed(
            "query",
            "--index",
            self.index,
            "--vector",
            "[0.1,0.1]",
            "--k",
            3,
            "--search-list-size",
            6,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["results"][0]["record_id"], 101)
        self.assertAlmostEqual(result["results"][0]["distance"], 2 ** 0.5 / 10,
                               places=5)

    def test_streaming_mode_loads_once_and_preserves_request_ids(self):
        self.build_index()
        requests = "\n".join(
            [
                json.dumps({"id": "near-origin", "vector": [0.2, 0.1]}),
                json.dumps({"id": "near-five", "vector": [5.1, 5.1]}),
            ]
        ) + "\n"
        completed = self.run_sembed(
            "query",
            "--index",
            self.index,
            "--stdin-jsonl",
            "--k",
            1,
            "--search-list-size",
            6,
            input_text=requests,
        )
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual([response["id"] for response in responses],
                         ["near-origin", "near-five"])
        self.assertEqual(responses[0]["results"][0]["record_id"], 101)
        self.assertEqual(responses[1]["results"][0]["record_id"], 104)

    def test_query_rejects_wrong_vector_dimensions(self):
        self.build_index()
        completed = self.run_sembed(
            "query",
            "--index",
            self.index,
            "--vector",
            "[1.0]",
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("query dimensions must match the index", completed.stderr)

    def test_full_search_returns_exact_distance_order(self):
        self.build_index()
        completed = self.run_sembed(
            "query",
            "--index",
            self.index,
            "--vector",
            "[0.2,0.1]",
            "--k",
            6,
            "--search-list-size",
            6,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(
            [neighbor["record_id"] for neighbor in result["results"]],
            [101, 102, 103, 104, 105, 106],
        )

    def test_index_build_is_deterministic(self):
        self.build_index()
        second = self.directory / "second.sembed"
        self.run_sembed(
            "index",
            "--dataset",
            self.dataset,
            "--output",
            second,
            "--degree-threshold",
            3,
            "--search-list-size",
            6,
        )
        self.assertEqual(self.index.read_bytes(), second.read_bytes())

    def test_index_with_trailing_bytes_is_rejected(self):
        self.build_index()
        trailing = self.directory / "trailing.sembed"
        trailing.write_bytes(self.index.read_bytes() + b"unexpected")
        completed = self.run_sembed(
            "query",
            "--index",
            trailing,
            "--vector",
            "[0,0]",
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("index file contains trailing data", completed.stderr)

    def test_invalid_build_configuration_is_rejected(self):
        completed = self.run_sembed(
            "index",
            "--dataset",
            self.dataset,
            "--output",
            self.index,
            "--degree-threshold",
            0,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("degree threshold must be positive", completed.stderr)

    def test_truncated_index_is_rejected(self):
        self.build_index()
        truncated = self.directory / "truncated.sembed"
        truncated.write_bytes(self.index.read_bytes()[:-7])
        completed = self.run_sembed(
            "query",
            "--index",
            truncated,
            "--vector",
            "[0,0]",
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("failed to read graph adjacency", completed.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sembed", type=Path, required=True)
    parsed, remaining = parser.parse_known_args()
    SEMBED_BINARY = parsed.sembed.resolve()
    unittest.main(argv=[__file__, *remaining])
