#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("run_benchmarks", SCRIPT)
BENCHMARKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARKS)


class BenchmarkMathTest(unittest.TestCase):
    def test_recall_at_k_ignores_duplicates_and_order(self):
        self.assertAlmostEqual(
            BENCHMARKS.recall_at_k([2, 2, 3, 5, 7], [2, 3, 4], 3),
            2.0 / 3.0,
        )

    def test_percentile_interpolates(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(BENCHMARKS.percentile(values, 0.50), 2.5)
        self.assertAlmostEqual(BENCHMARKS.percentile(values, 0.95), 3.85)

    def test_empty_percentile_is_zero(self):
        self.assertEqual(BENCHMARKS.percentile([], 0.95), 0.0)


class BenchmarkReportTest(unittest.TestCase):
    def test_comparison_matches_runs_with_the_same_workload(self):
        workload = {"query_count": 3, "k": 2, "seed": 42}
        runs = [
            {
                "name": "exact",
                "result": {
                    "algorithm": "bruteforce",
                    "dataset": {"path": "/dataset.bin"},
                    "workload": workload,
                    "metrics": {"queries_per_second": 10.0},
                },
            },
            {
                "name": "ann",
                "result": {
                    "algorithm": "vamana",
                    "dataset": {"path": "/dataset.bin"},
                    "workload": workload,
                    "metrics": {
                        "queries_per_second": 25.0,
                        "recall_at_k": 0.75,
                    },
                },
            },
        ]

        comparisons = BENCHMARKS.build_comparisons(runs)

        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]["run"], "ann")
        self.assertEqual(comparisons[0]["baseline"], "exact")
        self.assertEqual(
            comparisons[0]["queries_per_second_speedup_vs_bruteforce"], 2.5
        )
        self.assertEqual(
            comparisons[0]["recall_at_k_delta_vs_bruteforce"], -0.25
        )


if __name__ == "__main__":
    unittest.main()
