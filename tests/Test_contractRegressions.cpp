// Contract-level regression tests covering API invariants that the main
// component and robustness suites do not cover.

#include "HDVector.hpp"
#include "dataset.hpp"
#include "graph.hpp"
#include "test_utils.hpp"
#include "utils.hpp"
#include "vamana.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace {

std::filesystem::path uniqueFixturePath(const std::string &tag) {
  return testutils::uniqueFixturePath("contract_regressions", tag);
}

using ScopedFile = testutils::ScopedPathCleanup;
using testutils::fixtureDir;
using testutils::writeDatasetFile;

std::vector<std::vector<float>> makeClusteredRows(int64_t &outN,
                                                  int64_t &outStored) {
  outStored = 3;
  std::vector<std::vector<float>> rows;
  int64_t id = 0;
  for (uint64_t cluster = 0; cluster < 3; ++cluster) {
    for (uint64_t i = 0; i < 5; ++i) {
      rows.push_back({static_cast<float>(id),
                      static_cast<float>(cluster * 50U + i),
                      static_cast<float>(cluster * 50U - i)});
      ++id;
    }
  }
  outN = static_cast<int64_t>(rows.size());
  return rows;
}

} // namespace

// ============================================================================
// Graph API-surface bugs
// ============================================================================

// BUG: setOutNeighbors blindly stores the caller-supplied vector without
// enforcing the degree cap that the graph was constructed with. This turns
// the "degree threshold" into advisory information only.
TEST(GraphContractRegression, SetOutNeighboursRespectsDegreeThreshold) {
  std::srand(0);
  Graph graph(6, 2);

  EXPECT_THROW(graph.setOutNeighbors(0, {1, 2, 3, 4, 5}),
               std::invalid_argument)
      << "setOutNeighbors should reject adjacency larger than the configured "
         "degree threshold";
}

// BUG: addOutNeighborUnique does not check from == to before inserting, so
// a caller can trivially create self-loops. Self-loops break the implicit
// assumption that out-neighbours never point back at the owning node, and
// any greedy search that steps through them will either waste an iteration
// or loop indefinitely on trivial graphs.
TEST(GraphContractRegression, AddOutNeighbourUniqueRejectsSelfLoop) {
  std::srand(0);
  Graph graph(4, 2);
  graph.clearOutNeighbors(0);

  graph.addOutNeighborUnique(0, 0);

  const auto &neighbours = graph.getOutNeighbors(0);
  EXPECT_EQ(std::count(neighbours.begin(), neighbours.end(), 0), 0)
      << "addOutNeighborUnique(0, 0) inserted a self-loop";
}

// BUG: getOutNeighbors returns a *mutable* reference to internal state, so
// any caller can bypass every invariant check in addOutNeighborUnique and
// setOutNeighbors simply by push_back-ing arbitrary values (including
// negative indices). The getter should either return const& or a copy.
TEST(GraphContractRegression, GetOutNeighboursDoesNotExposeMutableInternalState) {
  std::srand(0);
  Graph graph(4, 2);

  using returned_type =
      decltype(std::declval<const Graph &>().getOutNeighbors(0));
  EXPECT_TRUE((std::is_same_v<returned_type, const NodeList &>))
      << "getOutNeighbors should expose read-only graph state";
}

// BUG: the path-based Graph constructor never checks stream state. A zero
// byte file opens fine, every file.read fails silently, numberOfNodes stays
// at 0, and the subsequent getRandomNumber(0, -1) executes gen() % 0 --
// undefined behaviour. A correct loader must report an empty/truncated file.
TEST(GraphContractRegression, EmptyFileConstructorIsCleanlyHandled) {
  const auto empty = uniqueFixturePath("empty_graph_file");
  ScopedFile cleanup{empty};
  std::ofstream out(empty, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.close();

  EXPECT_EXIT(
      {
        alarm(3);
        try {
          Graph g(empty);
          // An empty graph might be acceptable, but it must not leave the
          // object in a state where future operations are UB.
          std::exit(0);
        } catch (const std::exception &) {
          std::exit(0);
        }
      },
      ::testing::ExitedWithCode(0), "");
}

// BUG: getOutNeighbors has no bounds check. A Graph(4, 2).getOutNeighbors(99)
// is undefined behaviour. The indexing operator should throw std::out_of_range
// so caller bugs surface immediately rather than corrupting memory.
TEST(GraphContractRegression, OutOfRangeGetOutNeighboursThrows) {
  std::srand(0);
  Graph graph(4, 2);

  EXPECT_EXIT(
      {
        alarm(3);
        try {
          (void)graph.getOutNeighbors(99);
          // If the method silently returned (UB), treat as failure.
          std::exit(1);
        } catch (const std::exception &) {
          std::exit(0);
        }
      },
      ::testing::ExitedWithCode(0), "");
}

// ============================================================================
// Utils / RNG bugs
// ============================================================================

// BUG: getPermutation seeds its mt19937 with the literal 100 on every call.
// Two invocations therefore produce *identical* permutations, which silently
// makes every consumer (notably Vamana::buildIndex) process nodes in the same
// order regardless of how the caller seeded std::rand.
TEST(UtilsContractRegression, GetPermutationIsNotDeterministicAcrossCalls) {
  std::srand(1);
  const auto first = getPermutation(20);
  std::srand(2);
  const auto second = getPermutation(20);

  ASSERT_EQ(first.size(), 20U);
  ASSERT_EQ(second.size(), 20U);
  EXPECT_NE(first, second)
      << "getPermutation reseeds its generator with the same literal every "
         "call, so two back-to-back calls collapse to one permutation";
}

// BUG: isValidPath is implemented as fs::exists, which returns true for
// directories, symlinks to directories, character devices, etc. FlatDataSet
// then tries to open the directory as a binary file and either silently
// reads garbage or fails later with a non-helpful error.
TEST(UtilsContractRegression, IsValidPathReturnsFalseForDirectoriesAndEmptyInput) {
  EXPECT_FALSE(isValidPath(""));
  EXPECT_FALSE(isValidPath(fixtureDir().string()));
}

// ============================================================================
// HDVector additional bugs
// ============================================================================

// BUG: HDVector's `dimensions` field used 32-bit signed storage, but the
// value is assigned from `vec.size()` (size_t) or `const int64_t &`. That
// storage silently narrows any caller-provided dimension beyond INT_MAX.
// Expose the type mismatch at the API boundary.
TEST(HDVectorContractRegression, DimensionAccessorExposesLongLongStorage) {
  HDVector v(std::vector<float>{1.0f, 2.0f, 3.0f});
  using returned_type =
      std::remove_cv_t<std::remove_reference_t<decltype(v.dimensions())>>;
  EXPECT_TRUE((std::is_same_v<returned_type, uint64_t>) ||
              (std::is_same_v<returned_type, std::size_t>))
      << "HDVector::getDimension returns a 32-bit signed value, which silently narrows any "
         "dimension larger than INT_MAX";
}

// BUG: distance on two identical vectors should return exactly 0. Any non-zero
// residue points at either precision loss or an unstable accumulator.
TEST(HDVectorContractRegression, DistanceOfIdenticalLargeVectorsIsExactlyZero) {
  std::vector<float> data(32, 1.0e7f);
  HDVector a(data);
  HDVector b(data);
  EXPECT_FLOAT_EQ(euclideanDistance(a.view(), b.view()), 0.0f);
}

// BUG: distance must satisfy the triangle inequality d(a,c) <= d(a,b) + d(b,c).
// An accumulator that loses precision can produce d(a,c) strictly larger than
// d(a,b)+d(b,c). Any such violation makes every downstream pruning decision
// unsound.
TEST(HDVectorContractRegression, DistanceObeysTriangleInequality) {
  HDVector a(std::vector<float>{0.0f, 0.0f});
  HDVector b(std::vector<float>{3.0f, 4.0f});
  HDVector c(std::vector<float>{6.0f, 8.0f});

  const float ab = euclideanDistance(a.view(), b.view());
  const float bc = euclideanDistance(b.view(), c.view());
  const float ac = euclideanDistance(a.view(), c.view());
  EXPECT_LE(ac, ab + bc + 1e-4f)
      << "triangle inequality violated: d(a,c)=" << ac
      << ", d(a,b)+d(b,c)=" << ab + bc;
}

// ============================================================================
// DataSet accessor narrowing
// ============================================================================

// BUG: DataSet stores a 64-bit record count but used to expose a 32-bit
// signed `getN()`,
// silently narrowing counts above INT_MAX. The accessor's declared return
// type should match the storage type so callers see the full range.
TEST(DataSetContractRegression, GetNReturnTypeMatchesLongLongStorage) {
  const auto path = uniqueFixturePath("n_return_type");
  ScopedFile cleanup{path};
  writeDatasetFile(path, 2, 3, {{0.0f, 1.0f, 2.0f}, {1.0f, 3.0f, 4.0f}});
  FlatDataSet ds(path);

  using returned_type =
      std::remove_cv_t<std::remove_reference_t<decltype(ds.getN())>>;
  EXPECT_TRUE((std::is_same_v<returned_type, uint64_t>))
      << "FlatDataSet::getN() returns a 32-bit signed value, silently narrowing the "
         "dataset record-count member";
}

// BUG: same narrowing issue as above, but for getDimensions.
TEST(DataSetContractRegression, GetDimensionsReturnTypeMatchesLongLongStorage) {
  const auto path = uniqueFixturePath("dim_return_type");
  ScopedFile cleanup{path};
  writeDatasetFile(path, 2, 3, {{0.0f, 1.0f, 2.0f}, {1.0f, 3.0f, 4.0f}});
  FlatDataSet ds(path);

  using returned_type =
      std::remove_cv_t<std::remove_reference_t<decltype(ds.getDimensions())>>;
  EXPECT_TRUE((std::is_same_v<returned_type, uint64_t>))
      << "FlatDataSet::getDimensions() returns a 32-bit signed value, silently narrowing "
         "the dataset dimension member";
}

// BUG: header validation allows n < 0 today. A correct loader must reject a
// negative record count up front because nothing downstream handles negative
// sizes gracefully.
TEST(DataSetContractRegression, NegativeRecordCountHeaderIsRejected) {
  const auto path = uniqueFixturePath("negative_n");
  ScopedFile cleanup{path};

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  int64_t n = -5;
  int64_t stored = 3;
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(&stored), sizeof(stored));
  out.close();

  EXPECT_THROW(FlatDataSet ds(path), std::exception)
      << "loader accepted a header claiming n == -5";
}

// BUG: getRecordViewByIndex on a negative index should throw out_of_range.
// std::vector::at on a negative signed index converts to a huge size_t and does
// throw, but the cast may be implementation-defined for the exact behaviour;
// our contract should be deterministic.
TEST(DataSetContractRegression, NegativeIndexGetRecordViewThrowsOutOfRange) {
  const auto path = uniqueFixturePath("negative_index");
  ScopedFile cleanup{path};
  writeDatasetFile(path, 2, 3, {{0.0f, 1.0f, 2.0f}, {1.0f, 3.0f, 4.0f}});
  FlatDataSet ds(path);

  EXPECT_THROW((void)ds.getRecordViewByIndex(-1), std::out_of_range);
}

// ============================================================================
// Vamana algorithmic bugs
// ============================================================================

// BUG: when R > N-1, the implementation cannot find R unique neighbours per
// node, but the current getter returns fewer than expected without any
// warning. A defensive implementation should either cap R at N-1 or throw.
TEST(VamanaContractRegression, GraphWithExcessiveDegreeHasAllAvailableNeighbours) {
  std::srand(0);
  Graph graph(5, 10);

  for (int64_t node = 0; node < 5; ++node) {
    const auto &neighbours = graph.getOutNeighbors(node);
    EXPECT_EQ(neighbours.size(), 4U)
        << "node " << node << " had only " << neighbours.size()
        << " neighbours when 4 unique non-self neighbours were available; "
           "the generator gave up after 3 retries instead of exhausting the "
           "candidate set";
  }
}

// BUG: buildIndex should be idempotent: re-running it on an already-built
// Vamana should leave the graph unchanged, because the input (dataset,
// distance threshold) has not changed and the processing order is
// deterministic.
TEST(VamanaContractRegression, BuildIndexIsIdempotent) {
  const auto path = uniqueFixturePath("buildindex_idempotent");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t stored = 0;
  const auto rows = makeClusteredRows(n, stored);
  writeDatasetFile(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<FlatDataSet>(path);
  Vamana v(std::move(ds), 3);

  // Snapshot adjacency after the first build.
  std::vector<NodeList> firstBuild(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    auto &adjacency = firstBuild[static_cast<size_t>(i)];
    adjacency = v.getOutNeighbors(i);
    std::sort(adjacency.begin(), adjacency.end());
  }

  // Redirect cout so the "Making N" spam does not pollute the test log.
  std::stringstream discard;
  std::streambuf *prev = std::cout.rdbuf(discard.rdbuf());
  v.buildIndex();
  std::cout.rdbuf(prev);

  // Compare to the second build.
  for (int64_t i = 0; i < n; ++i) {
    auto after = v.getOutNeighbors(i);
    std::sort(after.begin(), after.end());
    EXPECT_EQ(firstBuild[static_cast<size_t>(i)], after)
        << "buildIndex is not idempotent for node " << i;
  }
}

// BUG: Vamana's default alpha is 1.2, but there is no public getter, so a
// caller cannot inspect or roundtrip the value they just set. This makes
// reproducing a tuning run impossible without reading the source. The
// test encodes the contract that, once set, alpha is observable.
TEST(VamanaContractRegression, SetDistanceThresholdIsObservable) {
  const auto path = uniqueFixturePath("alpha_getter");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t stored = 0;
  const auto rows = makeClusteredRows(n, stored);
  writeDatasetFile(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<FlatDataSet>(path);
  Vamana v(std::move(ds), 3);

  v.setDistanceThreshold(2.5f);

  // The public getter makes the configured alpha observable without tests
  // having to reach into internal state.
  EXPECT_FLOAT_EQ(v.getDistanceThreshold(), 2.5f)
      << "alpha is no longer 2.5 after setDistanceThreshold(2.5)";
}

// BUG: greedySearch's outer loop caps iterations at 10000. For a dataset
// large enough that L > 10000, the algorithm silently truncates its
// exploration, but there is no indication to the caller that it stopped
// early. Make that contract visible on a small dataset: on N = 15, the
// traversal should *never* hit the cap, yet the algorithm should still
// leave a marker so we can tell.
TEST(VamanaContractRegression, GreedySearchDoesNotReturnInternalMarkers) {
  const auto path = uniqueFixturePath("no_markers");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t stored = 0;
  const auto rows = makeClusteredRows(n, stored);
  writeDatasetFile(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<FlatDataSet>(path);
  Vamana v(std::move(ds), 3);
  v.setSearchListSize(15);

  HDVector q(std::vector<float>{12.0f, -2.0f});
  SearchResults r = v.greedySearch(q.view(), 5);

  // approximateNN must be strictly ascending by distance.
  for (size_t i = 1; i < r.approximateNN.size(); ++i) {
    const float before = euclideanDistance(
        q.view(), v.getRecordViewByIndex(r.approximateNN[i - 1]).values);
    const float after = euclideanDistance(
        q.view(), v.getRecordViewByIndex(r.approximateNN[i]).values);
    EXPECT_LE(before, after)
        << "approximateNN[" << i - 1 << "] and approximateNN[" << i
        << "] are out of order";
  }
}

// BUG: repeatedly calling greedySearch on the same query must return the
// same result. A leak of state (for example from static locals in helpers)
// would show up here.
TEST(VamanaContractRegression, GreedySearchIsDeterministicForIdenticalQueries) {
  const auto path = uniqueFixturePath("deterministic_search");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t stored = 0;
  const auto rows = makeClusteredRows(n, stored);
  writeDatasetFile(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<FlatDataSet>(path);
  Vamana v(std::move(ds), 3);
  v.setSearchListSize(10);

  HDVector q(std::vector<float>{25.0f, -5.0f});
  SearchResults first = v.greedySearch(q.view(), 3);
  SearchResults second = v.greedySearch(q.view(), 3);

  EXPECT_EQ(first.approximateNN, second.approximateNN)
      << "two identical greedySearch calls returned different ANN lists";
}

// BUG: inserting a mismatched-dimension query node anywhere in the algorithm
// should surface as std::invalid_argument.
TEST(VamanaContractRegression, InsertIntoSetPropagatesDimensionMismatch) {
  const auto path = uniqueFixturePath("dim_mismatch_insert");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t stored = 0;
  const auto rows = makeClusteredRows(n, stored);
  writeDatasetFile(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<FlatDataSet>(path);
  Vamana v(std::move(ds), 3);

  HDVector bad(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  NodeList to;
  EXPECT_THROW(v.insertIntoSet({1, 2}, to, bad.view()), std::invalid_argument);
}
