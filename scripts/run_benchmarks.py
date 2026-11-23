#!/usr/bin/env python3

import argparse
import json
import random
import struct
import subprocess
import time
from pathlib import Path

np = None


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark sembed.")
    parser.add_argument("--sembed-binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_dataset(path):
    with path.open("rb") as source:
        rows, columns = struct.unpack("=qq", source.read(16))
        dimensions = columns - 1
        record_dtype = np.dtype(
            [("id", "=i8"), ("vector", "=f4", (dimensions,))], align=False
        )
        records = np.fromfile(source, dtype=record_dtype, count=rows)
        return np.ascontiguousarray(records["vector"]).reshape(rows, dimensions)


def percentile(values, fraction):
    if not values:
        return 0.0
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (
        position - lower
    )


def exact_query(vectors, query, k, excluded_node=None):
    distances = np.sum((vectors - query) ** 2, axis=1)
    if excluded_node is not None:
        distances[excluded_node] = np.inf
    return np.argsort(distances, kind="stable")[:k].tolist()


def recall_at_k(approximate, exact, k):
    return len(set(approximate[:k]) & set(exact[:k])) / min(k, len(exact))


def measurements(latencies, recalls):
    return {
        "recall_at_k": sum(recalls) / len(recalls),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "queries_per_second": len(latencies) * 1000 / sum(latencies),
    }


def result(run, dataset, vectors, metrics):
    return {
        "algorithm": run["algorithm"],
        "dataset": {
            "path": str(dataset),
        },
        "workload": {
            "query_count": run.get("query_count", 100),
            "k": run.get("k", 10),
            "seed": run.get("seed", 0),
        },
        "metrics": metrics,
    }


def run_bruteforce(run, dataset, vectors, queries):
    k = run.get("k", 10)
    latencies = []
    for node in queries:
        started = time.perf_counter_ns()
        exact_query(vectors, vectors[node], k, node)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    return result(run, dataset, vectors, measurements(latencies, [1.0] * len(queries)))


def ask(process, vector, request_id):
    process.stdin.write(json.dumps({"id": request_id, "vector": vector.tolist()}) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())


def run_vamana(run, binary, dataset, vectors, queries, artifact_dir):
    k = run.get("k", 10)
    degree = run.get("degree_threshold", 64)
    search_list = run.get("search_list_size", 100)
    alpha = run.get("distance_threshold", 1.2)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    index = artifact_dir / "index.sembed"

    started = time.perf_counter()
    subprocess.run(
        [str(binary), "index", "--dataset", str(dataset), "--output", str(index),
         "--degree-threshold", str(degree), "--search-list-size", str(search_list),
         "--distance-threshold", str(alpha)],
        check=True,
    )
    build_seconds = time.perf_counter() - started

    process = subprocess.Popen(
        [str(binary), "query", "--index", str(index), "--stdin-jsonl", "--k",
         str(k + 1), "--search-list-size", str(search_list)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ask(process, vectors[queries[0]], "warmup")
    expected = [exact_query(vectors, vectors[node], k, node) for node in queries]
    approximate = []
    latencies = []
    for position, node in enumerate(queries):
        started = time.perf_counter_ns()
        response = ask(process, vectors[node], position)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        nodes = [neighbor["node"] for neighbor in response["results"]]
        nodes.remove(node)
        approximate.append(nodes[:k])
    process.stdin.close()
    process.wait()

    recalls = [recall_at_k(found, exact, k) for found, exact in zip(approximate, expected)]
    output = result(run, dataset, vectors, measurements(latencies, recalls))
    output["metrics"]["build_time_seconds"] = build_seconds
    return output


def comparison_key(result):
    workload = result["workload"]
    return result["dataset"]["path"], workload["query_count"], workload["k"], workload["seed"]


def build_comparisons(runs):
    baselines = {
        comparison_key(run["result"]): run for run in runs
        if run["result"]["algorithm"] == "bruteforce"
    }
    comparisons = []
    for run in runs:
        benchmark = run["result"]
        baseline = baselines.get(comparison_key(benchmark))
        if benchmark["algorithm"] == "vamana" and baseline:
            comparisons.append({
                "run": run["name"],
                "baseline": baseline["name"],
                "queries_per_second_speedup_vs_bruteforce": (
                    benchmark["metrics"]["queries_per_second"]
                    / baseline["result"]["metrics"]["queries_per_second"]
                ),
                "recall_at_k_delta_vs_bruteforce": benchmark["metrics"]["recall_at_k"] - 1.0,
            })
    return comparisons


def main():
    global np
    import numpy
    np = numpy

    args = parse_args()
    profile = json.loads(args.config.read_text())
    runs = []
    for run in profile["runs"]:
        dataset = (args.build_dir / run["dataset"]).resolve()
        vectors = load_dataset(dataset)
        count = min(run.get("query_count", 100), len(vectors))
        queries = random.Random(run.get("seed", 0)).sample(range(len(vectors)), count)
        if run["algorithm"] == "bruteforce":
            output = run_bruteforce(run, dataset, vectors, queries)
        else:
            output = run_vamana(
                run, args.sembed_binary.resolve(), dataset, vectors, queries,
                args.output.parent / "artifacts" / run["name"],
            )
        runs.append({"name": run["name"], "result": output})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "runs": runs,
        "comparisons": build_comparisons(runs),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
