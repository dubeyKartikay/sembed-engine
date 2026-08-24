#!/usr/bin/env python3

import argparse
import json
import math
import random
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

np = None


def load_numpy():
    global np
    try:
        import numpy
    except ImportError as error:
        raise SystemExit(
            "run_benchmarks.py requires NumPy; install it with "
            "'python3 -m pip install numpy'"
        ) from error
    np = numpy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark sembed from Python against an exact NumPy baseline."
    )
    parser.add_argument("--sembed-binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_dataset(path):
    started = time.perf_counter()
    with path.open("rb") as source:
        header = source.read(16)
        if len(header) != 16:
            raise ValueError(f"truncated dataset header: {path}")
        records, stored_dimensions = struct.unpack("=qq", header)
        if records < 0 or stored_dimensions <= 1:
            raise ValueError(f"invalid dataset shape: {path}")
        dimensions = stored_dimensions - 1
        record_ids = np.fromfile(source, dtype=np.int64, count=records)
        values = np.fromfile(
            source, dtype=np.float32, count=records * dimensions
        )
        if record_ids.size != records or values.size != records * dimensions:
            raise ValueError(f"truncated dataset payload: {path}")
        if source.read(1):
            raise ValueError(f"dataset contains trailing bytes: {path}")
    vectors = values.reshape((records, dimensions))
    return record_ids, vectors, time.perf_counter() - started


def resolve_input_path(raw_path, build_dir, config_dir):
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    config_relative = (config_dir / candidate).resolve()
    if config_relative.exists():
        return config_relative
    return (build_dir / candidate).resolve()


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def exact_query(vectors, query, k, excluded_node=None):
    distances = np.sum((vectors - query) ** 2, axis=1)
    if excluded_node is not None:
        distances[excluded_node] = np.inf
    limit = min(k, vectors.shape[0] - (1 if excluded_node is not None else 0))
    if limit <= 0:
        return []
    candidates = np.argpartition(distances, limit - 1)[:limit]
    ranked = candidates[np.lexsort((candidates, distances[candidates]))]
    return [int(node) for node in ranked]


def recall_at_k(approximate, exact, k):
    denominator = min(k, len(exact))
    if denominator == 0:
        return 1.0
    return len(set(approximate[:k]).intersection(exact[:k])) / denominator


def workload(vectors, query_vectors, count, seed, exclude_self):
    available = query_vectors.shape[0]
    query_count = min(count or 100, available)
    indices = random.Random(seed).sample(range(available), query_count)
    return [
        {
            "index": index,
            "vector": query_vectors[index],
            "excluded_node": index if exclude_self else None,
        }
        for index in indices
    ]


def metrics(latencies_ms, recalls, build_seconds=None, index_path=None,
            restart_seconds=None, load_seconds=None, visited_nodes=None):
    total_seconds = sum(latencies_ms) / 1000.0
    return {
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 1.0,
        "latency_p50_ms": percentile(latencies_ms, 0.50),
        "latency_p95_ms": percentile(latencies_ms, 0.95),
        "queries_per_second": (
            len(latencies_ms) / total_seconds if total_seconds > 0 else 0.0
        ),
        "build_time_seconds": build_seconds,
        "ram_footprint_bytes": None,
        "ssd_footprint_bytes": index_path.stat().st_size if index_path else None,
        "restart_time_seconds": restart_seconds,
        "insert_throughput_vectors_per_second": None,
        "dataset_load_time_seconds": load_seconds,
        "query_dataset_load_time_seconds": None,
        "average_visited_nodes": visited_nodes,
    }


def run_bruteforce(run, vectors, queries, dataset_path, load_seconds):
    k = run.get("k", 10)
    latencies = []
    for query in queries:
        started = time.perf_counter_ns()
        exact_query(
            vectors, query["vector"], k, query["excluded_node"]
        )
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {
        "algorithm": "bruteforce",
        "dataset": {
            "path": str(dataset_path),
            "size": vectors.shape[0],
            "dimensions": vectors.shape[1],
        },
        "workload": {
            "query_dataset_path": None,
            "query_count": len(queries),
            "k": k,
            "seed": run.get("seed", 0),
            "exclude_self": run.get("exclude_self", True),
        },
        "metrics": metrics(
            latencies, [1.0] * len(queries), load_seconds=load_seconds
        ),
        "notes": "Exact baseline and all benchmark aggregation run in Python/NumPy.",
    }


class QueryProcess:
    def __init__(self, binary, index_path, k, search_list_size):
        self.process = subprocess.Popen(
            [
                str(binary),
                "query",
                "--index",
                str(index_path),
                "--stdin-jsonl",
                "--k",
                str(k),
                "--search-list-size",
                str(search_list_size),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def query(self, vector, request_id):
        request = {"id": request_id, "vector": vector.tolist()}
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read()
            raise RuntimeError(f"sembed query process exited early: {error}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def close(self):
        self.process.stdin.close()
        return_code = self.process.wait()
        error = self.process.stderr.read()
        if return_code != 0:
            raise RuntimeError(f"sembed query process failed: {error}")


def run_vamana(run, binary, vectors, queries, dataset_path, artifact_dir,
               load_seconds):
    k = run.get("k", 10)
    degree = run.get("degree_threshold", 64)
    search_list = run.get("search_list_size", 100)
    alpha = run.get("distance_threshold", 1.2)
    index_path = artifact_dir / "index.sembed"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    build_command = [
        str(binary),
        "index",
        "--dataset",
        str(dataset_path),
        "--output",
        str(index_path),
        "--degree-threshold",
        str(degree),
        "--search-list-size",
        str(search_list),
        "--distance-threshold",
        str(alpha),
    ]
    started = time.perf_counter()
    subprocess.run(build_command, check=True, capture_output=True, text=True)
    build_seconds = time.perf_counter() - started

    requested_k = k + 1 if run.get("exclude_self", True) else k
    started = time.perf_counter()
    process = QueryProcess(binary, index_path, requested_k, search_list)
    if queries:
        process.query(queries[0]["vector"], "warmup")
    restart_seconds = time.perf_counter() - started

    exact_results = [
        exact_query(vectors, query["vector"], k, query["excluded_node"])
        for query in queries
    ]
    latencies = []
    approximate_results = []
    try:
        for position, query in enumerate(queries):
            started_ns = time.perf_counter_ns()
            response = process.query(query["vector"], position)
            latencies.append(
                (time.perf_counter_ns() - started_ns) / 1_000_000.0
            )
            nodes = [item["node"] for item in response["results"]]
            excluded = query["excluded_node"]
            if excluded is not None:
                nodes = [node for node in nodes if node != excluded]
            approximate_results.append(nodes[:k])
    finally:
        process.close()

    recalls = [
        recall_at_k(approximate, exact, k)
        for approximate, exact in zip(approximate_results, exact_results)
    ]
    return {
        "algorithm": "vamana",
        "dataset": {
            "path": str(dataset_path),
            "size": vectors.shape[0],
            "dimensions": vectors.shape[1],
        },
        "workload": {
            "query_dataset_path": None,
            "query_count": len(queries),
            "k": k,
            "seed": run.get("seed", 0),
            "exclude_self": run.get("exclude_self", True),
        },
        "configuration": {
            "degree_threshold": degree,
            "search_list_size": search_list,
            "distance_threshold": alpha,
        },
        "metrics": metrics(
            latencies,
            recalls,
            build_seconds=build_seconds,
            index_path=index_path,
            restart_seconds=restart_seconds,
            load_seconds=load_seconds,
        ),
        "notes": "Python owns the workload, exact baseline, timing, and aggregation; sembed only builds and queries the index.",
    }


def build_comparisons(runs):
    baselines = {}
    for run in runs:
        result = run["result"]
        if result["algorithm"] == "bruteforce":
            key = (
                result["dataset"]["path"],
                result["workload"]["query_count"],
                result["workload"]["k"],
                result["workload"]["seed"],
            )
            baselines[key] = run

    comparisons = []
    for run in runs:
        result = run["result"]
        if result["algorithm"] != "vamana":
            continue
        key = (
            result["dataset"]["path"],
            result["workload"]["query_count"],
            result["workload"]["k"],
            result["workload"]["seed"],
        )
        baseline = baselines.get(key)
        if baseline:
            comparisons.append(
                {
                    "run": run["name"],
                    "baseline": baseline["name"],
                    "queries_per_second_speedup_vs_bruteforce": (
                        result["metrics"]["queries_per_second"]
                        / baseline["result"]["metrics"]["queries_per_second"]
                    ),
                    "recall_at_k_delta_vs_bruteforce": (
                        result["metrics"]["recall_at_k"] - 1.0
                    ),
                }
            )
    return comparisons


def main():
    load_numpy()
    args = parse_args()
    binary = args.sembed_binary.resolve()
    profile = json.loads(args.config.read_text(encoding="utf-8"))
    build_dir = args.build_dir.resolve()
    config_dir = args.config.resolve().parent
    output = args.output.resolve()
    artifact_root = output.parent / "artifacts"
    output.parent.mkdir(parents=True, exist_ok=True)

    loaded = {}
    runs = []
    for run in profile["runs"]:
        dataset_path = resolve_input_path(run["dataset"], build_dir, config_dir)
        if dataset_path not in loaded:
            loaded[dataset_path] = load_dataset(dataset_path)
        _, vectors, load_seconds = loaded[dataset_path]
        queries = workload(
            vectors,
            vectors,
            run.get("query_count", 0),
            run.get("seed", 0),
            run.get("exclude_self", True),
        )
        if run["algorithm"] in ("bruteforce", "brute-force"):
            result = run_bruteforce(
                run, vectors, queries, dataset_path, load_seconds
            )
        elif run["algorithm"] == "vamana":
            result = run_vamana(
                run,
                binary,
                vectors,
                queries,
                dataset_path,
                artifact_root / run["name"],
                load_seconds,
            )
        else:
            raise ValueError(f"unknown algorithm: {run['algorithm']}")
        runs.append({"name": run["name"], "result": result})

    report = {
        "profile": str(args.config.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "comparisons": build_comparisons(runs),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
