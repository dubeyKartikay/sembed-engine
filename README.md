# sembed-engine

`sembed-engine` is a small C++17 approximate-nearest-neighbor engine for dense
float embeddings. It builds a Vamana-style graph index and searches it with a
full query vector.

The project deliberately has only two C++ product targets:

- `sembed-lib` builds the reusable core as `libsembed` and exports the CMake
  target `sembed::sembed`.
- `sembed` is a thin command-line wrapper over that library.

Benchmark workloads, exact baselines, timing, aggregation, and reports live in
Python rather than in the engine.

## Build

Requirements:

- CMake 3.21 or newer
- A C++17 compiler
- Python 3 for integration tests and benchmark orchestration
- NumPy only when running benchmarks

Configure, build, and test:

```sh
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The build products are:

```text
build/libsembed.a  # reusable static library; filename varies by platform
build/sembed       # CLI
build/sembed-tests # native regression test executable; not installed/exported
```

CLI11 and nlohmann/json are vendored header-only dependencies of the CLI. The
core library uses the C++ standard library and does not expose either dependency
through its public interface.

The native GoogleTest suite is always built. It covers the core index and the
current CLI workflow, and invokes the dependency-free Python benchmark tests.
CTest also runs the Python public-API integration suite. Tests were adapted to
the current architecture instead of restoring the removed C++ benchmark
harness, Armadillo adapter, or experimental clustering implementation.

## Reusable C++ API

The convenience umbrella header is:

```cpp
#include <sembed/sembed.hpp>
```

Larger consumers can include only the semantic area they use:

- `<sembed/dataset_view.hpp>` for borrowed row-major datasets
- `<sembed/vector_view.hpp>` for borrowed query vectors
- `<sembed/index.hpp>` for index configuration, construction, and persistence
- `<sembed/query.hpp>` for query configuration and ANN results

`<sembed/sembed.hpp>` only includes those focused headers; it does not define
additional API types.

Build from an in-memory row-major dataset and query it:

```cpp
#include <sembed/sembed.hpp>

#include <cstdint>
#include <vector>

std::vector<int64_t> ids = {10, 20, 30};
std::vector<float> vectors = {
    0.0F, 0.0F,
    1.0F, 0.0F,
    0.0F, 1.0F,
};

sembed::DatasetView dataset{
    ids.data(), vectors.data(), /*size=*/3, /*dimensions=*/2};
sembed::IndexConfig indexConfig{
    /*degreeThreshold=*/2,
    /*searchListSize=*/3,
    /*distanceThreshold=*/1.2F,
};

sembed::Index index = sembed::buildIndex(dataset, indexConfig);
index.save("vectors.sembed");

std::vector<float> queryVector = {0.1F, 0.1F};
std::vector<sembed::Neighbor> neighbors = sembed::query(
    index,
    sembed::FloatVectorView(queryVector),
    sembed::QueryConfig{/*k=*/2, /*searchListSize=*/3});
```

You can also build from the repository's binary dataset format:

```cpp
sembed::Index index = sembed::buildIndex("vectors.bin", indexConfig);
```

Load a previously saved index without the original dataset:

```cpp
sembed::Index index = sembed::Index::load("vectors.sembed");
```

Each returned `sembed::Neighbor` contains:

- `node`: the internal zero-based graph node
- `recordId`: the caller-provided dataset record ID
- `distance`: Euclidean distance from the query vector

A query search-list size of zero reuses the value stored in the index.

### Importing with CMake

Install the package:

```sh
cmake --install build --prefix /desired/prefix
```

Consume it from another CMake project:

```cmake
find_package(sembed CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE sembed::sembed)
```

The installed package contains the public header, library, CLI, and CMake
package metadata.

## CLI

The CLI exposes the same two core operations.

Build a self-contained index:

```sh
./build/sembed index \
  --dataset ./build/gvec.bin \
  --degree-threshold 32 \
  --search-list-size 64 \
  --distance-threshold 1.2 \
  --output ./build/gvec.sembed
```

Query with a complete vector:

```sh
./build/sembed query \
  --index ./build/gvec.sembed \
  --vector '[0.1,0.2,0.3]' \
  --k 10 \
  --search-list-size 64
```

The vector dimensionality must exactly match the index. Results are emitted as
JSON.

For benchmarks and other high-throughput callers, load the index once and
stream JSON Lines through stdin:

```sh
printf '%s\n' \
  '{"id":"q1","vector":[0.1,0.2,0.3]}' \
  '{"id":"q2","vector":[0.4,0.5,0.6],"k":5}' \
  | ./build/sembed query \
      --index ./build/gvec.sembed \
      --stdin-jsonl \
      --k 10 \
      --search-list-size 64
```

The process keeps the index resident and emits one JSON response per input
line. An optional request `id` is copied into the response.

## Persistence

A `.sembed` index is self-contained and versioned. It stores:

- record IDs and dense float vectors
- graph adjacency
- medoid and degree threshold
- build search-list size and distance threshold

The original dataset is not needed after the index is saved. Loading validates
the magic, version, dataset shape, graph bounds, adjacency constraints,
truncation, and trailing bytes.

The input dataset format remains:

```text
int64 record_count
int64 stored_dimensions       # vector dimensions + 1
int64 record_ids[record_count]
float vectors[record_count][stored_dimensions - 1]
```

All numeric fields use the host's native binary representation, so this input
format and the current index format are intended for the same architecture.

## Python benchmarks

Generate the checked-in embedding fixtures when needed:

```sh
python3 scripts/generate_embedding_fixtures.py --output-dir build
```

Install NumPy and run the smoke profile:

```sh
python3 -m pip install numpy
python3 scripts/run_benchmarks.py \
  --sembed-binary ./build/sembed \
  --config ./benchmarks/local_smoke.json \
  --build-dir ./build \
  --output ./build/benchmark-report/local-smoke.json
```

The Python driver owns query sampling, the exact NumPy baseline, recall,
latency, throughput, comparisons, and report generation. It communicates with
the long-lived CLI JSONL protocol so the index is loaded once per run.

## Sanitizers

AddressSanitizer and UndefinedBehaviorSanitizer remain opt-in library build
settings:

```sh
cmake -S . -B build-asan -DSEMBED_ENABLE_ASAN=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
```
