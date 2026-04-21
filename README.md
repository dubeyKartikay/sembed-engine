# sembed-engine

`sembed-engine` is a small C++17 approximate nearest neighbor engine for dense vector embeddings.
It provides:

- Binary dataset loaders for embedding collections stored on disk.
- `HDVector`, a dense floating-point vector type with Euclidean distance.
- A Vamana-style graph index for approximate nearest neighbor search.
- Graph persistence so an index can be saved and loaded later.
- A batch stochastic k-means helper for coarse clustering experiments.

The main reusable APIs live under [`src/include`](src/include), while the `sembed` executable is a minimal demo target.

## What The Project Does

At a high level, the engine turns a dataset of dense embeddings into a bounded-degree directed graph. Each node is an embedding vector, and edges point to useful neighbors. Queries are answered by walking that graph greedily instead of scanning the full dataset.

That gives you:

- Faster nearest-neighbor lookup than brute-force search on larger datasets.
- Deterministic index construction and tests for the same inputs.
- A simple persistence format for graph reuse across runs.
- A compact API for experiments with embeddings, ANN search, and clustering.

## Requirements

You need:

- CMake 3.21 or newer
- A C++17 compiler
- A C compiler for the fixture converters used during the build
- Python 3
- A CMake generator such as `Ninja` or `Unix Makefiles`

## Build

Configure and compile:

```sh
cmake -S . -B build
cmake --build build
```

This build produces:

- `build/libutils.a`
- `build/libbatch_stocastic_kmeans.a`
- `build/Test`
- `build/sembed`

The build also generates deterministic embedding fixtures used by tests:

- `build/gvec.bin`
- `build/w2v.bin`
- `build/gvec.words.bin`
- `build/w2v.words.bin`

Those fixtures are generated automatically by [`scripts/generate_embedding_fixtures.py`](scripts/generate_embedding_fixtures.py) from the checked-in subsets in [`testdata/embeddings`](testdata/embeddings).

## Test

Run the full suite with CTest:

```sh
ctest --test-dir build --output-on-failure
```

Run the GoogleTest binary directly:

```sh
./build/Test
```

Generate the JUnit-style test report configured by CMake:

```sh
cmake --build build --target test_report
```

That writes:

- `build/test-report/junit.xml`

As of the current CMake configuration, there is a `test_report` target, but there are no `test_html_report` or `test_html_report_open` targets.

## Public API Overview

The main headers are:

- [`src/include/HDVector.hpp`](src/include/HDVector.hpp)
- [`src/include/dataset.hpp`](src/include/dataset.hpp)
- [`src/include/graph.hpp`](src/include/graph.hpp)
- [`src/include/vamana.hpp`](src/include/vamana.hpp)
- [`src/include/batch_stocastic_kmeans.hpp`](src/include/batch_stocastic_kmeans.hpp)

### 1. Loading a dataset

There are two dataset implementations:

- `FileDataSet`: reads records from disk on demand
- `InMemoryDataSet`: loads the full dataset into memory up front

Example:

```cpp
#include <memory>
#include "dataset.hpp"

int main() {
  auto dataset = std::make_unique<InMemoryDataSet>("build/gvec.bin");

  const uint64_t n = dataset->getN();
  const uint64_t d = dataset->getDimentions();

  RecordView record = dataset->getRecordViewByIndex(0);
  int64_t id = record.recordId;
  float first_value = (*record.vector)[0];

  auto batch = dataset->getNRecordViewsFromIndex(10, 5);
  auto vectors = dataset->getNHDVectorsFromIndex(10, 5);
}
```

Key methods:

- `getN()`: number of records
- `getDimentions()`: vector dimensionality, excluding the stored record id
- `getRecordViewByIndex(i)`: fetch one record
- `getNRecordViewsFromIndex(i, count)`: fetch a contiguous range
- `getNHDVectorsFromIndex(i, count)`: fetch only the vectors from a range

### 2. Vector math with `HDVector`

`HDVector` stores a dense `std::vector<float>` and exposes:

- construction from dimension count
- construction from an existing `std::vector<float>`
- bounds-checked indexing via `operator[]`
- Euclidean distance with `HDVector::distance(a, b)`

Example:

```cpp
#include "HDVector.hpp"

HDVector a(std::vector<float>{1.0f, 2.0f});
HDVector b(std::vector<float>{4.0f, 6.0f});

float d = HDVector::distance(a, b);  // 5.0
```

### 3. Building and querying a Vamana index

Construct an index from a dataset:

```cpp
#include <memory>
#include "dataset.hpp"
#include "vamana.hpp"

auto dataset = std::make_unique<InMemoryDataSet>("build/gvec.bin");
HDVector query = *dataset->getRecordViewByIndex(42).vector;

Vamana index(std::move(dataset), /*R=*/64, /*alpha=*/1.2f);

index.setSeachListSize(100);

SearchResults result = index.greedySearch(query, /*k=*/10);

for (NodeId node : result.approximateNN) {
  RecordView neighbor = index.m_dataSet->getRecordViewByIndex(node);
}
```

Important knobs:

- `R` / degree threshold: max out-neighbors retained per node
- `alpha` / distance threshold: pruning aggressiveness during graph construction
- `m_searchListSize` via `setSeachListSize(L)`: candidate list size during greedy search

Important methods:

- `buildIndex()`: rebuild the graph from the dataset
- `greedySearch(query, k)`: search using an arbitrary `HDVector`
- `search(queryNode, k)`: search using an existing dataset node as the query
- `save(path)`: persist the graph

`SearchResults` contains:

- `approximateNN`: the final top-`k` candidate node ids
- `visited`: nodes expanded during graph traversal

### 4. Saving and loading a graph index

Save an already-built graph:

```cpp
index.save("my-index.graph");
```

Load a persisted graph without rebuilding:

```cpp
auto dataset = std::make_unique<InMemoryDataSet>("build/gvec.bin");
Vamana loaded(std::move(dataset), std::filesystem::path("my-index.graph"));
```

You can also supply an existing `Graph` directly:

```cpp
Graph graph("my-index.graph");
auto dataset = std::make_unique<InMemoryDataSet>("build/gvec.bin");
Vamana loaded(std::move(dataset), graph);
```

### 5. Clustering with batch stochastic k-means

The project also includes a simple batch clustering utility:

```cpp
#include "batch_stocastic_kmeans.hpp"
#include "dataset.hpp"

InMemoryDataSet dataset("build/gvec.bin");
std::vector<Cluster> clusters = clusterize_data(dataset, /*k=*/8, /*iterations=*/25);
```

Each `Cluster` contains:

- `center`: a `Point`
- `Points`: the assigned points

The center update is not a raw arithmetic centroid output. The implementation computes the mean, then snaps the center back to the closest real point in the cluster.

## Binary Formats

### Dataset format

Both `FileDataSet` and `InMemoryDataSet` read the same binary layout:

```text
Header:
  int64_t n
  int64_t stored_dimensions

For each record:
  int64_t record_id
  float values[stored_dimensions - 1]
```

Notes:

- `stored_dimensions` includes the record id field.
- The exposed vector dimensionality is `stored_dimensions - 1`.
- Record ids are preserved as `int64_t`.

So if you have 50-dimensional embeddings, `stored_dimensions` must be `51`.

### Graph format

`Graph::save()` writes:

```text
Header:
  uint64_t node_count
  uint64_t degree_threshold
  uint64_t mediod_or_sentinel

For each node:
  uint64_t degree
  uint64_t neighbors[degree]
```

If no medoid is stored, the graph file uses `UINT64_MAX` as the sentinel value.

## How ANN Search Works

This implementation follows the broad structure of a Vamana-style navigable graph index.

### Distance metric

All search and pruning decisions use Euclidean distance:

$$
d(x, y) = \sqrt{\sum_{i=1}^{D}(x_i - y_i)^2}
$$

where:

- `D` is the embedding dimensionality
- `x` and `y` are embedding vectors

The implementation computes this in `HDVector::distance`.

### Graph construction

The index stores one graph node per record. Construction starts from an initial random bounded-degree graph and then refines it by processing the dataset in a deterministic random permutation.

For each point `p`:

1. Run greedy graph search using `p` as the query.
2. Collect visited nodes as candidate neighbors.
3. Prune the candidate set so only strong edges remain.
4. Add reciprocal backlinks where appropriate.
5. Re-prune neighbors whose degree grows past `R`.

The result is a sparse directed graph designed to be easy to navigate with local search.

### Greedy search

Search begins from the graph medoid and repeatedly expands the best currently known unvisited candidate. The candidate set is maintained in ascending order of distance to the query vector and truncated to size `L = m_searchListSize`.

Conceptually:

1. Start from the medoid.
2. Insert its outgoing neighbors into the candidate set.
3. Always expand the closest unvisited candidate next.
4. Stop when there are no more expandable candidates inside the top `L`.
5. Return the closest `k` candidates found.

This is why the key practical parameters are:

- `R`: graph sparsity / branching factor
- `L`: how much of the frontier search keeps
- `k`: how many results you want returned

Larger `L` usually improves recall, but it also increases search work.

### Alpha-pruning math

The pruning rule is what keeps the graph sparse without throwing away too much navigability.

When the algorithm considers a candidate neighbor `p'` while building the out-neighbors of node `p`, it compares `p'` against an already accepted neighbor `p*`.

`p'` is pruned if:

$$
\alpha \cdot d(p^\*, p') \le d(p, p')
$$

where:

- `p` is the node being assigned neighbors
- `p*` is a selected neighbor already kept
- `p'` is another candidate neighbor
- `\alpha` is the distance threshold, exposed in code as `m_distanceThreshold`

Interpretation:

- If `p'` is already well-covered by `p*`, then the direct edge `p -> p'` is less useful.
- A larger `alpha` usually keeps more edges.
- A smaller `alpha` prunes more aggressively.

This is the core geometric tradeoff in the index: keep enough edges for navigability, but not so many that search becomes dense and expensive.

## How The K-Means Helper Works

The clustering helper is a simple iterative batch procedure:

1. Choose `k` initial centers from random dataset records.
2. Assign each point to its nearest center.
3. Compute the mean of each cluster.
4. Replace that mean with the real cluster point closest to the mean.
5. Repeat for the requested number of iterations.

Mathematically, the mean for cluster `C` is:

$$
\mu_C = \frac{1}{|C|}\sum_{x \in C} x
$$

But the implementation does not keep `\mu_C` directly as the center. It selects:
$$
c_C = \arg\min_{x \in C} d(x, \mu_C)
$$

That makes the center an actual dataset point rather than a synthetic floating-point centroid.

## Using It From Another CMake Project

If you want to consume the code as a subdirectory:

```cmake
add_subdirectory(sembed-engine)

target_link_libraries(my_app PRIVATE utils batch_stocastic_kmeans)
target_include_directories(my_app PRIVATE
  ${CMAKE_SOURCE_DIR}/sembed-engine/src/include
)
```

Then include the public headers from `src/include`.


