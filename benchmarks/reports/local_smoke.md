# Local Smoke Benchmark Report

This report is the checked-in output for the deterministic smoke profile in
[`benchmarks/local_smoke.json`](../local_smoke.json).

Generated from:

```sh
python3 ./scripts/run_benchmarks.py \
  --benchmark-binary ./build/sembed_benchmark \
  --config ./benchmarks/local_smoke.json \
  --build-dir ./build \
  --output ./build/benchmark-report/local-smoke.json
```

Snapshot from the current run:

| dataset | algorithm | recall@10 | p50 latency ms | p95 latency ms | build s | RAM bytes | index bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gvec.bin` | brute force | 1.00 | 0.27 | 0.43 | - | 598016 | - |
| `gvec.bin` | vamana r32 l64 | 1.00 | 22.98 | 24.64 | 10.37 | 1204224 | 49080 |
| `w2v.bin` | brute force | 1.00 | 1.49 | 1.85 | - | 847872 | - |
| `w2v.bin` | vamana r32 l64 | 1.00 | 142.27 | 144.33 | 100.48 | 1978368 | 67592 |

Notes:

- These are fixture-scale results on deterministic 256-record datasets.
- The report is intended for regression tracking and transparency, not for
  claiming production competitiveness on tiny datasets.
- On these fixtures, brute force is faster than the current Vamana
  implementation while recall remains identical.
