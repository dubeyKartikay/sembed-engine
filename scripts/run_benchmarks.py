#!/usr/bin/env python3

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a reproducible benchmark profile against sembed_benchmark."
    )
    parser.add_argument(
        "--benchmark-binary",
        type=Path,
        required=True,
        help="Path to the built sembed_benchmark executable.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON benchmark profile to execute.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Build directory used to resolve generated fixture paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the combined JSON report should be written.",
    )
    return parser.parse_args()


def load_profile(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_input_path(raw_path: str, build_dir: Path, config_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    config_relative = (config_dir / candidate).resolve()
    if config_relative.exists():
        return config_relative
    return (build_dir / candidate).resolve()


def run_single_benchmark(
    run_config,
    benchmark_binary: Path,
    build_dir: Path,
    config_dir: Path,
    artifact_root: Path,
):
    run_name = run_config["name"]
    artifact_dir = artifact_root / run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(benchmark_binary.resolve()),
        "--algorithm",
        run_config["algorithm"],
        "--dataset",
        str(resolve_input_path(run_config["dataset"], build_dir, config_dir)),
        "--artifact-dir",
        str(artifact_dir),
    ]

    optional_args = {
        "--query-dataset": run_config.get("query_dataset"),
        "--dataset-mode": run_config.get("dataset_mode"),
        "--query-count": run_config.get("query_count"),
        "--k": run_config.get("k"),
        "--seed": run_config.get("seed"),
        "--degree-threshold": run_config.get("degree_threshold"),
        "--search-list-size": run_config.get("search_list_size"),
        "--distance-threshold": run_config.get("distance_threshold"),
    }
    for flag, value in optional_args.items():
        if value is None:
            continue
        if flag == "--query-dataset":
            value = resolve_input_path(value, build_dir, config_dir)
        command.extend([flag, str(value)])

    if "exclude_self" in run_config:
        command.extend(
            ["--exclude-self", "true" if run_config["exclude_self"] else "false"]
        )

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        if error.returncode < 0:
            signal_number = -error.returncode
            raise SystemExit(
                f"{run_name} was terminated by signal {signal_number} while "
                f"running {benchmark_binary}. Endpoint protection can cause this."
            ) from error
        raise

    return {"name": run_name, "result": json.loads(completed.stdout)}


def build_comparisons(runs):
    brute_force_by_dataset = {}
    for run in runs:
        result = run["result"]
        if result["algorithm"] != "bruteforce":
            continue
        key = (
            result["dataset"]["path"],
            result["workload"]["query_dataset_path"],
            result["workload"]["query_count"],
            result["workload"]["k"],
        )
        brute_force_by_dataset[key] = run

    comparisons = []
    for run in runs:
        result = run["result"]
        if result["algorithm"] == "bruteforce":
            continue

        key = (
            result["dataset"]["path"],
            result["workload"]["query_dataset_path"],
            result["workload"]["query_count"],
            result["workload"]["k"],
        )
        baseline = brute_force_by_dataset.get(key)
        if baseline is None:
            continue

        baseline_metrics = baseline["result"]["metrics"]
        current_metrics = result["metrics"]
        speedup = None
        if baseline_metrics["queries_per_second"]:
            speedup = (
                current_metrics["queries_per_second"]
                / baseline_metrics["queries_per_second"]
            )

        comparisons.append(
            {
                "run": run["name"],
                "baseline": baseline["name"],
                "queries_per_second_speedup_vs_bruteforce": speedup,
                "recall_at_k_delta_vs_bruteforce": current_metrics["recall_at_k"]
                - baseline_metrics["recall_at_k"],
            }
        )

    return comparisons


def main():
    args = parse_args()
    profile = load_profile(args.config)
    build_dir = args.build_dir.resolve()
    config_dir = args.config.resolve().parent
    report_dir = args.output.resolve().parent
    artifact_root = report_dir / "artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    runs = []
    for run_config in profile["runs"]:
        runs.append(
            run_single_benchmark(
                run_config,
                args.benchmark_binary,
                build_dir,
                config_dir,
                artifact_root,
            )
        )

    report = {
        "profile": str(args.config.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "comparisons": build_comparisons(runs),
    }

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
