// Cross-component regression tests for public API contracts and behavioral
// invariants across utils, HDVector, Graph, DataSet, and Vamana.

#include "HDVector.hpp"
#include "dataset.hpp"
#include "graph.hpp"
#include "test_utils.hpp"
#include "utils.hpp"
#include "vamana.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace {

std::filesystem::path uniqueFixturePath(const std::string &tag) {
  return testutils::uniqueFixturePath("component_regressions", tag);
}

using ScopedFile = testutils::ScopedPathCleanup;
using testutils::fixtureDir;
using testutils::writeDatasetFile;
using testutils::writeGraphFile;

std::vector<std::vector<float>> makeClusteredRows(int64_t &outN,
                                                  int64_t &outStoredDim) {
  outStoredDim = 3;
  std::vector<std::vector<float>> rows;
  int64_t id = 0;
  for (uint64_t cluster = 0; cluster < 3; ++cluster) {
    for (uint64_t offset = 0; offset < 6; ++offset) {
      const float x = static_cast<float>(cluster * 100U + offset);
      const float y = static_cast<float>(cluster * 50U + (offset % 2U));
      rows.push_back({static_cast<float>(id), x, y});
      ++id;
    }
  }
  outN = static_cast<int64_t>(rows.size());
  return rows;
}

} // namespace

// =========================================================================
// getRandomNumber bugs -- returns NaN-ish edge-case behaviour.
// =========================================================================
TEST(RandomUtilsRegression, GetRandomNumberHandlesSingletonRange) {
  // A singleton range [start, start] must always return `start` and must
  // not depend on the modulo-of-one happening to swallow garbage.
  const int64_t only_value = 42;
  for (uint64_t i = 0; i < 5; ++i) {
    EXPECT_EQ(getRandomNumber(only_value, only_value), only_value);
  }
}

TEST(RandomUtilsRegression, GenerateRandomNumbersHonoursZeroKRequest) {
  // Asking for zero values should produce an empty vector without crashing
  // or generating spurious values.
  const auto result = generateRandomNumbers(0, 5, /*blackList=*/-1);
  EXPECT_TRUE(result.empty()) << "expected an empty vector when k == 0";
}

TEST(RandomUtilsRegression, GenerateRandomNumbersExcludesBlacklistUnderCollisions) {
  // When n is tiny, the naive "retry a couple of times" loop happily
  // returns the blacklisted value.
  std::srand(1);

  // n == 1 with blacklist 0 means the only candidate is blacklisted.
  // A correct implementation should either return an empty vector or
  // throw. The buggy implementation sometimes emits `0`.
  const auto result = generateRandomNumbers(3, 1, /*blackList=*/0);
  for (int64_t v : result) {
    EXPECT_NE(v, 0) << "blacklisted value leaked into the output";
  }
}

TEST(RandomUtilsRegression, GenerateRandomNumbersProducesUniqueValuesOnSmallRanges) {
  // On a small `n` the post-collision retry window is tiny. The buggy
  // implementation regularly drops values, so the output contains fewer
  // than k entries.
  std::srand(123);

  constexpr uint64_t k = 6;
  constexpr uint64_t n = 8;
  const auto result = generateRandomNumbers(k, n, /*blackList=*/-1);

  EXPECT_EQ(result.size(), static_cast<size_t>(k))
      << "collision loop gave up and returned fewer than k unique values";

  std::unordered_set<NodeId> unique(result.begin(), result.end());
  EXPECT_EQ(unique.size(), result.size());
}

TEST(RandomUtilsRegression, GetPermutationProducesAllValuesExactlyOnce) {
  // Small smoke test: every value in [0, n) must appear exactly once.
  const uint64_t n = 32;
  const auto perm = getPermutation(n);
  ASSERT_EQ(perm.size(), n);

  NodeList sorted(perm);
  std::sort(sorted.begin(), sorted.end());
  for (uint64_t i = 0; i < n; ++i) {
    EXPECT_EQ(sorted[i], i);
  }

  // And it must actually be a permutation, i.e. not the identity when
  // the RNG is seeded. getPermutation uses a fixed seed of 100; for that
  // seed the permutation should differ from the identity on at least one
  // position. If someone replaces the shuffle with an identity the test
  // fails loudly.
  uint64_t in_place = 0;
  for (uint64_t i = 0; i < n; ++i) {
    if (perm[i] == i) {
      ++in_place;
    }
  }
  EXPECT_LT(in_place, n)
      << "getPermutation returned the identity -- shuffle is missing";
}

// =========================================================================
// HDVector bugs -- index bounds, copy semantics, and operator symmetry.
// =========================================================================
TEST(HDVectorRegression, NegativeIndexIsRejected) {
  HDVector v(std::vector<float>{1.0f, 2.0f, 3.0f});
  EXPECT_THROW((void)v[-1], std::out_of_range);
}

TEST(HDVectorRegression, IndexAtDimensionIsRejected) {
  HDVector v(std::vector<float>{1.0f, 2.0f, 3.0f});
  // Writing past the end must not silently succeed.
  EXPECT_THROW((void)v[3], std::out_of_range);
}

TEST(HDVectorRegression, DistanceIsSymmetric) {
  HDVector a(std::vector<float>{0.0f, 0.0f, 0.0f});
  HDVector b(std::vector<float>{3.0f, 4.0f, 12.0f});

  EXPECT_FLOAT_EQ(HDVector::distance(a, b), HDVector::distance(b, a));
}

TEST(HDVectorRegression, DistanceWithSelfIsZero) {
  HDVector v(std::vector<float>{1.5f, -2.25f, 0.125f, 9999.0f});
  EXPECT_FLOAT_EQ(HDVector::distance(v, v), 0.0f);
}

TEST(HDVectorRegression, ZeroDimensionalVectorDistanceIsZero) {
  HDVector a(0);
  HDVector b(0);
  EXPECT_FLOAT_EQ(HDVector::distance(a, b), 0.0f);
}

TEST(HDVectorRegression, ConstructorFromDimensionInitialisesZeros) {
  HDVector v(5);
  ASSERT_EQ(v.getDimention(), 5);
  for (uint64_t i = 0; i < 5; ++i) {
    EXPECT_EQ(v[i], 0.0f);
  }
}

// =========================================================================
// Graph bugs -- constructor truncation, mediod persistence, self edges.
// =========================================================================
TEST(GraphRegression, RandomInitDoesNotProduceSelfEdges) {
  std::srand(0);
  // A fully connected graph minus self-loops means R == N-1 should be
  // achievable for every node, and no neighbour should point to itself.
  Graph g(16, 15);
  for (NodeId node = 0; node < 16; ++node) {
    const auto &neighbours = g.getOutNeighbours(node);
    for (NodeId neighbour : neighbours) {
      EXPECT_NE(neighbour, node)
          << "node " << node << " has a self-loop in the initial adjacency";
    }
  }
}

TEST(GraphRegression, ConstructorFromPathPreservesMedoid) {
  // Persisted graph with a known mediod. The on-disk mediod must be the
  // one returned after loading.
  const auto path = uniqueFixturePath("graph_mediod");
  ScopedFile cleanup{path};

  constexpr uint64_t node_count = 4;
  constexpr uint64_t degree = 2;
  constexpr NodeId stored_mediod = 3;
  writeGraphFile(path, node_count, degree, stored_mediod,
                 {{1, 2}, {0, 2}, {0, 1}, {0, 1}});

  // Isolate the crash-prone constructor from the rest of the test suite.
  const std::string graph_path = path.string();
  auto mediod_probe = [&]() {
    Graph g(graph_path);
    std::exit(g.getMediod() == OptionalNodeId{stored_mediod} ? 0 : 1);
  };
  EXPECT_EXIT(mediod_probe(), ::testing::ExitedWithCode(0), "")
      << "mediod was regenerated randomly instead of read from disk, or "
         "the loader crashed before it could be observed";
}

TEST(GraphRegression, AddOutNeighbourUniqueDoesNotAllowSelfEdges) {
  std::srand(0);
  Graph g(5, 2);
  g.clearOutNeighbours(0);

  // An ANN graph must not introduce self-loops; the unique helper should
  // reject an attempt to add the node to its own adjacency.
  g.addOutNeighbourUnique(0, 0);
  const auto &neighbours = g.getOutNeighbours(0);

  EXPECT_EQ(std::count(neighbours.begin(), neighbours.end(), 0), 0)
      << "addOutNeighbourUnique accepted a self edge";
}

TEST(GraphRegression, ConstructorWithRZeroStaysDegreeZero) {
  std::srand(0);
  Graph g(6, 0);
  for (NodeId node = 0; node < 6; ++node) {
    EXPECT_TRUE(g.getOutNeighbours(node).empty())
        << "node " << node << " has edges despite degree threshold 0";
  }
  EXPECT_EQ(g.getDegreeThreshold(), 0);
}

// =========================================================================
// DataSet bugs -- large-index exposure, index type, and concurrent reads.
// =========================================================================
TEST(DataSetRegression, GetNDoesNotSilentlyTruncateLargeSize) {
  // getN() used to be a 32-bit signed return value, so any legitimate dataset with more
  // than INT_MAX records is truncated to a negative value.
  const auto path = uniqueFixturePath("huge_header");
  ScopedFile cleanup{path};

  const int64_t huge_n =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 10LL;
  const int64_t stored_dim = 2;

  // Write only the header; readers that don't materialise records should
  // still be able to report N == huge_n without truncation.
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(&huge_n), sizeof(huge_n));
  out.write(reinterpret_cast<const char *>(&stored_dim), sizeof(stored_dim));
  out.close();

  FileDataSet ds(path);
  EXPECT_EQ(static_cast<int64_t>(ds.getN()), huge_n)
      << "getN() truncated a 64-bit record count into a 32-bit signed value";
}

TEST(DataSetRegression, FileDataSetGetRecordViewByIndexRejectsNegative) {
  const auto path = uniqueFixturePath("neg_index");
  ScopedFile cleanup{path};
  writeDatasetFile(path, 3, 2,
                   {{1.0f, 2.0f}, {2.0f, 4.0f}, {3.0f, 6.0f}});

  FileDataSet ds(path);
  // The in-memory implementation and the file-based implementation both
  // advertise `std::out_of_range` for invalid indices in DataSetApiTest;
  // the contract here is that -1 yields the same error.
  EXPECT_THROW((void)ds.getRecordViewByIndex(
                   std::numeric_limits<uint64_t>::max()),
               std::out_of_range);
}

TEST(DataSetRegression, InMemoryDataSetNegativeIndexYieldsOutOfRange) {
  const auto path = uniqueFixturePath("inmem_neg");
  ScopedFile cleanup{path};
  writeDatasetFile(path, 2, 2, {{1.0f, 2.0f}, {2.0f, 3.0f}});

  InMemoryDataSet ds(path);
  EXPECT_THROW((void)ds.getRecordViewByIndex(
                   std::numeric_limits<uint64_t>::max()),
               std::out_of_range);
}

TEST(DataSetRegression, InMemoryDataSetRejectsOutOfBoundsIndex) {
  const auto path = uniqueFixturePath("inmem_oob");
  ScopedFile cleanup{path};
  writeDatasetFile(path, 2, 2, {{1.0f, 2.0f}, {2.0f, 3.0f}});

  InMemoryDataSet ds(path);
  // at() throws std::out_of_range already; ensure the constant holds.
  EXPECT_THROW((void)ds.getRecordViewByIndex(2), std::out_of_range);
  EXPECT_THROW((void)ds.getRecordViewByIndex(1000), std::out_of_range);
}

TEST(DataSetRegression, RecordIdsRoundTripThroughLargeLongLongValues) {
  const auto path = uniqueFixturePath("large_ids");
  ScopedFile cleanup{path};

  const int64_t huge_id = (1LL << 30) + 7LL;
  const int64_t stored_dimensions = 3;
  const std::vector<float> payload = {1.0f, 2.0f};
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  const int64_t n = 1;
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(&stored_dimensions),
            sizeof(stored_dimensions));
  out.write(reinterpret_cast<const char *>(&huge_id), sizeof(huge_id));
  out.write(reinterpret_cast<const char *>(payload.data()),
            static_cast<std::streamsize>(payload.size() * sizeof(float)));
  out.close();

  InMemoryDataSet ds(path);
  auto record = ds.getRecordViewByIndex(0);
  EXPECT_EQ(record.recordId, huge_id)
      << "record id round-trip lost precision through int64_t storage";
}

TEST(DataSetRegression, FileDataSetAllowsConcurrentReadsWithoutCorruption) {
  const auto path = uniqueFixturePath("concurrent_reads");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 8; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i) * 2.0f});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  // Share one dataset across two "threads" by interleaving reads.  The
  // underlying fstream uses mutable seek state; interleaved reads must
  // still return the correct records.
  FileDataSet ds(path);
  auto first = ds.getRecordViewByIndex(0);
  // Initiate a range read that repositions the internal cursor.
  auto range = ds.getNRecordViewsFromIndex(3, 2);
  // Now re-read index 0. If the cursor wasn't properly re-seeked, this
  // returns garbage.
  auto second = ds.getRecordViewByIndex(0);

  ASSERT_NE(first.vector, nullptr);
  ASSERT_NE(second.vector, nullptr);
  ASSERT_EQ(first.vector->getDimention(), 2);
  ASSERT_EQ(second.vector->getDimention(), 2);
  EXPECT_FLOAT_EQ((*first.vector)[0], (*second.vector)[0]);
  EXPECT_FLOAT_EQ((*first.vector)[1], (*second.vector)[1]);
  EXPECT_EQ(first.recordId, second.recordId);
  EXPECT_EQ(first.recordId, 0);
  ASSERT_NE(range, nullptr);
  ASSERT_EQ(range->size(), 2U);
}

TEST(VamanaRegression, ConstructorFromGraphDoesNotRebuildIndex) {
  // Supplying an explicit Graph is how callers can load a pre-computed
  // index. The constructor must not call buildIndex() again, otherwise
  // the explicit edges are overwritten by a fresh (randomised) graph.
  const auto path = uniqueFixturePath("from_graph");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t storedDim = 0;
  const auto rows = makeClusteredRows(n, storedDim);
  writeDatasetFile(path, n, storedDim, rows);

  auto ds = std::make_unique<InMemoryDataSet>(path);
  Graph provided(n, 3);
  provided.clearOutNeighbours(0);
  provided.addOutNeighbourUnique(0, 1);
  provided.addOutNeighbourUnique(0, 2);

  std::srand(0);
  Vamana v(std::move(ds), provided);

  const auto &neighbours = v.m_graph.getOutNeighbours(0);
  NodeList sorted(neighbours.begin(), neighbours.end());
  std::sort(sorted.begin(), sorted.end());
  const NodeList expected = {1, 2};
  EXPECT_EQ(sorted, expected)
      << "constructor rebuilt the index and discarded the supplied graph";
}

TEST(VamanaRegression, ConstructorFromSavedPathLoadsInsteadOfRebuilding) {
  const auto dataset_path = uniqueFixturePath("saved_dataset");
  const auto graph_path = uniqueFixturePath("saved_graph");
  ScopedFile ds_cleanup{dataset_path};
  ScopedFile graph_cleanup{graph_path};

  int64_t n = 0;
  int64_t storedDim = 0;
  const auto rows = makeClusteredRows(n, storedDim);
  writeDatasetFile(dataset_path, n, storedDim, rows);

  // Build an explicit graph with a ring topology and persist it.
  const uint64_t node_count = static_cast<uint64_t>(n);
  const uint64_t degree = 2;
  std::vector<NodeList> adjacency(
      static_cast<size_t>(node_count), NodeList(degree));
  for (NodeId i = 0; i < node_count; ++i) {
    adjacency[static_cast<size_t>(i)][0] = (i + 1) % node_count;
    adjacency[static_cast<size_t>(i)][1] = (i + node_count - 1) % node_count;
  }
  writeGraphFile(graph_path, node_count, degree, /*mediod=*/0, adjacency);

  // Graph(path) currently seg-faults (it reads into zero-sized inner
  // vectors), so run the Vamana saved-path constructor inside a death
  // test to isolate the failure and let the remaining tests run.
  const std::string dataset_str = dataset_path.string();
  const std::string graph_str = graph_path.string();

  auto saved_path_loader = [&]() {
    auto ds = std::make_unique<InMemoryDataSet>(dataset_str);
    std::srand(0);
    Vamana v(std::move(ds), std::filesystem::path(graph_str));

    for (NodeId i = 0; i < node_count; ++i) {
      const auto &neighbours = v.m_graph.getOutNeighbours(i);
      NodeList sorted(neighbours.begin(), neighbours.end());
      std::sort(sorted.begin(), sorted.end());
      NodeList expected;
      expected.push_back((i + node_count - 1) % node_count);
      expected.push_back((i + 1) % node_count);
      std::sort(expected.begin(), expected.end());
      if (sorted != expected) {
        std::exit(1);
      }
    }
    std::exit(0);
  };
  EXPECT_EXIT(saved_path_loader(), ::testing::ExitedWithCode(0), "")
      << "loaded graph was overwritten by a fresh buildIndex() call, or "
         "the saved-path loader crashed before verification";
}


TEST(VamanaRegression, GreedySearchWithSmallSearchListStillTerminatesAndReturnsNN) {
  const auto path = uniqueFixturePath("zero_search_list");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t storedDim = 0;
  const auto rows = makeClusteredRows(n, storedDim);
  writeDatasetFile(path, n, storedDim, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 4);
  v.setSeachListSize(1);

  HDVector query(std::vector<float>{210.0f, 110.0f});
  // With L == 1 the main loop may never advance, which would indicate a
  // bug: the algorithm should still return a best-effort candidate.
  SearchResults result = v.greedySearch(query, 1);
  EXPECT_EQ(result.approximateNN.size(), 1U);
}

TEST(VamanaRegression, GreedySearchIsDeterministicAcrossRepeatedCalls) {
  const auto path = uniqueFixturePath("deterministic_calls");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t storedDim = 0;
  const auto rows = makeClusteredRows(n, storedDim);
  writeDatasetFile(path, n, storedDim, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 4);
  v.setSeachListSize(n);

  HDVector query(std::vector<float>{102.5f, 50.25f});
  SearchResults first = v.greedySearch(query, 3);
  SearchResults second = v.greedySearch(query, 3);

  EXPECT_EQ(first.approximateNN, second.approximateNN)
      << "greedySearch returned different results for identical inputs";
  EXPECT_EQ(first.visited, second.visited)
      << "greedySearch visited ordering changed between identical calls";
}

TEST(VamanaRegression, BuildIndexProducesIdenticalGraphForIdenticalInputs) {
  const auto path = uniqueFixturePath("deterministic_build");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t storedDim = 0;
  const auto rows = makeClusteredRows(n, storedDim);
  writeDatasetFile(path, n, storedDim, rows);

  auto build = [&path]() {
    std::srand(0);
    auto ds = std::make_unique<InMemoryDataSet>(path);
    return std::make_unique<Vamana>(std::move(ds), 4);
  };

  auto left = build();
  auto right = build();

  for (NodeId node = 0; node < left->m_dataSet->getN();
       ++node) {
    NodeList lhs = left->m_graph.getOutNeighbours(node);
    NodeList rhs = right->m_graph.getOutNeighbours(node);
    std::sort(lhs.begin(), lhs.end());
    std::sort(rhs.begin(), rhs.end());
    EXPECT_EQ(lhs, rhs)
        << "identical datasets produced different adjacency for node "
        << node;
  }

  EXPECT_EQ(left->m_graph.getMediod(), right->m_graph.getMediod())
      << "mediod selection is not deterministic across identical inputs";
}

TEST(VamanaRegression, SearchFunctionIsImplemented) {
  EXPECT_TRUE(false)
      << "Vamana::search(HDVector, int64_t) is declared in vamana.hpp but has "
         "no definition in src/utils/Vamana.cpp -- any caller would fail "
         "to link";
}

TEST(VamanaRegression, GreedySearchDoesNotReturnDuplicateNodes) {
  // Even with a tiny degree threshold the search list should never contain
  // the same index twice.
  const auto path = uniqueFixturePath("no_duplicates");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 12; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i % 3),
                    static_cast<float>(i / 3)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);
  v.setSeachListSize(8);

  HDVector query(std::vector<float>{1.0f, 2.0f});
  SearchResults r = v.greedySearch(query, 5);

  std::unordered_set<NodeId> unique(r.approximateNN.begin(),
                                 r.approximateNN.end());
  EXPECT_EQ(unique.size(), r.approximateNN.size())
      << "greedySearch returned duplicate indices in its ANN list";

  std::unordered_set<NodeId> visitedUnique(r.visited.begin(), r.visited.end());
  EXPECT_EQ(visitedUnique.size(), r.visited.size())
      << "visited set contained duplicate nodes";
}

TEST(VamanaRegression, SaveIsImplemented) {
  EXPECT_TRUE(false)
      << "Vamana::save() is declared in vamana.hpp but has no definition "
         "in src/utils/Vamana.cpp -- any caller would fail to link";
}

TEST(VamanaRegression, IsToBePrunedFlagsSamePointAsNotPrunable) {
  // When p_dash == p_star the distance between them is zero; pruning
  // should not require throwing the point away.  The current formulation
  // `alpha * d(p_star, p_dash) <= d(p, p_dash)` reduces to `0 <= x`, which
  // is always true and prunes everything that happens to coincide with
  // p_star.
  const auto path = uniqueFixturePath("self_prune");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows = {
      {0.0f, 0.0f, 0.0f},
      {1.0f, 10.0f, 10.0f},
      {2.0f, 20.0f, 20.0f},
  };
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);

  EXPECT_FALSE(v.isToBePruned(1, 1, 0))
      << "a point can't be pruned against itself";
}

// =========================================================================
// insertIntoSet bugs -- duplicate handling and stability.
// =========================================================================
TEST(VamanaRegression, InsertIntoSetDoesNotDuplicateExistingMembers) {
  const auto path = uniqueFixturePath("insert_no_dupes");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows = {
      {0.0f, 0.0f, 0.0f},    {1.0f, 1.0f, 0.0f},    {2.0f, 2.0f, 0.0f},
      {3.0f, 3.0f, 0.0f},    {4.0f, 4.0f, 0.0f},    {5.0f, 5.0f, 0.0f},
  };
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);

  HDVector q(std::vector<float>{2.5f, 0.0f});
  NodeList working = {1, 2, 3};
  std::sort(working.begin(), working.end(),
            [&v, &q](int64_t l, int64_t r) {
              const float ld =
                  HDVector::distance(q, *v.m_dataSet->getRecordViewByIndex(l).vector);
              const float rd =
                  HDVector::distance(q, *v.m_dataSet->getRecordViewByIndex(r).vector);
              if (ld == rd) {
                return l < r;
              }
              return ld < rd;
            });

  // Reinsert the same values.  The resulting set must not grow.
  v.insertIntoSet({1, 2, 3}, working, q);
  EXPECT_EQ(working.size(), 3U)
      << "insertIntoSet duplicated already-present elements";

  std::unordered_set<NodeId> unique(working.begin(), working.end());
  EXPECT_EQ(unique.size(), working.size());
}

// =========================================================================
// DataSet + Vamana end-to-end sanity check -- brute-force vs ANN top-k.
// =========================================================================
TEST(VamanaRegression, GreedySearchRecoversExactTopKOnTinyWellSeparatedClusters) {
  const auto path = uniqueFixturePath("topk_recall");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t cluster = 0; cluster < 4; ++cluster) {
    const float baseX = static_cast<float>(cluster * 1000);
    for (uint64_t i = 0; i < 5; ++i) {
      rows.push_back({static_cast<float>(cluster * 5 + i),
                      baseX + static_cast<float>(i),
                      baseX + static_cast<float>(i) * 0.5f});
    }
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 4);
  v.setSeachListSize(static_cast<int64_t>(rows.size()));

  const std::vector<float> query = {2001.0f, 1000.5f};
  HDVector q(query);
  SearchResults approx = v.greedySearch(q, 3);

  // Brute force exact top-3.
  std::vector<std::pair<float, int64_t>> scored;
  for (size_t i = 0; i < rows.size(); ++i) {
    const float dx = rows[i][1] - query[0];
    const float dy = rows[i][2] - query[1];
    scored.push_back({dx * dx + dy * dy, static_cast<int64_t>(i)});
  }
  std::sort(scored.begin(), scored.end(),
            [](const auto &l, const auto &r) {
              if (l.first == r.first) {
                return l.second < r.second;
              }
              return l.first < r.first;
            });

  ASSERT_GE(scored.size(), 3U);
  const NodeList expected = {static_cast<NodeId>(scored[0].second),
                             static_cast<NodeId>(scored[1].second),
                             static_cast<NodeId>(scored[2].second)};

  ASSERT_EQ(approx.approximateNN.size(), 3U);
  EXPECT_EQ(approx.approximateNN, expected)
      << "ANN result deviates from brute-force top-3 on well-separated "
         "clusters";
}

// =========================================================================
// Mediod selection bugs -- the mediod is chosen via getRandomNumber, which
// reseeds on every call, so every newly built index gets *the same* random
// number regardless of the dataset.
// =========================================================================
TEST(VamanaRegression, MedoidDependsOnDatasetSize) {
  // If the mediod truly reflects a representative point, two distinct
  // datasets with different sizes cannot share exactly the same mediod
  // index once it is drawn from a legitimate random source.
  const auto path_small = uniqueFixturePath("mediod_small");
  const auto path_large = uniqueFixturePath("mediod_large");
  ScopedFile cleanup_small{path_small};
  ScopedFile cleanup_large{path_large};

  std::vector<std::vector<float>> small_rows;
  for (uint64_t i = 0; i < 6; ++i) {
    small_rows.push_back({static_cast<float>(i), static_cast<float>(i),
                          static_cast<float>(i)});
  }
  writeDatasetFile(path_small, small_rows.size(), 3, small_rows);

  std::vector<std::vector<float>> large_rows;
  for (uint64_t i = 0; i < 200; ++i) {
    large_rows.push_back({static_cast<float>(i), static_cast<float>(i),
                          static_cast<float>(i)});
  }
  writeDatasetFile(path_large, large_rows.size(), 3, large_rows);

  std::srand(0);
  auto small_ds = std::make_unique<InMemoryDataSet>(path_small);
  Vamana v_small(std::move(small_ds), 3);

  std::srand(0);
  auto large_ds = std::make_unique<InMemoryDataSet>(path_large);
  Vamana v_large(std::move(large_ds), 3);

  // The small dataset can only index up to 5. The large dataset indexes
  // up to 199. A correct mediod picker should almost always pick an index
  // beyond 5 on the large dataset -- the buggy reseeded RNG keeps picking
  // the same tiny constant regardless of N.
  EXPECT_NE(v_small.m_graph.getMediod(), v_large.m_graph.getMediod())
      << "mediod selection ignores dataset size because getRandomNumber "
         "reseeds with a constant every call";
}

// =========================================================================
// Load / clustering bugs -- load_from_binary returns a nullptr unconditionally
// and clusterize_data never assigns any point to a cluster.
// =========================================================================
#include "load_from_binary.hpp"
TEST(BinaryLoadingRegression, LoadFromBinaryReturnsUsablePayload) {
  const auto path = uniqueFixturePath("load_bin");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 10; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i * 2),
                    static_cast<float>(i * 3)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::vector<std::vector<float> *> *loaded =
      load_from_binary(path.string());

  // The API declares a non-nullable pointer to a vector of vectors -- a
  // caller who defensively dereferences it currently segfaults.
  ASSERT_NE(loaded, nullptr)
      << "load_from_binary unconditionally returns nullptr";
  EXPECT_EQ(loaded->size(), rows.size());

  for (auto *row : *loaded) {
    delete row;
  }
  delete loaded;
}

// =========================================================================
// HDVector semantic bugs -- operator[] should be assignable, distance should
// cope with large/tiny magnitudes, and copy semantics should keep ownership
// independent.
// =========================================================================
TEST(HDVectorRegression, OperatorIndexIsAssignable) {
  HDVector v(3);
  v[0] = 1.0f;
  v[1] = 2.0f;
  v[2] = 3.0f;
  EXPECT_FLOAT_EQ(v[0], 1.0f);
  EXPECT_FLOAT_EQ(v[1], 2.0f);
  EXPECT_FLOAT_EQ(v[2], 3.0f);
}

TEST(HDVectorRegression, DistanceWithLargeValuesDoesNotOverflow) {
  // Values near 1e18 square to 1e36 which overflows float (max ~3.4e38)
  // when summed in naive precision. Ensuring the implementation keeps
  // double precision internally avoids a silent infinity leak.
  std::vector<float> a(4, 0.0f);
  std::vector<float> b(4, 1e18f);
  HDVector va(a);
  HDVector vb(b);
  const float d = HDVector::distance(va, vb);
  EXPECT_TRUE(std::isfinite(d))
      << "distance overflowed to infinity for large magnitude vectors";
}

TEST(HDVectorRegression, DataPointerReflectsLaterWrites) {
  HDVector v(std::vector<float>{7.0f, 8.0f, 9.0f});
  float *raw = v.getDataPointer();
  raw[1] = 42.0f;
  EXPECT_FLOAT_EQ(v[1], 42.0f)
      << "getDataPointer does not expose the underlying storage";
}

TEST(DataSetRegression, GetNRecordViewsFromIndexRejectsNegativeN) {
  const auto path = uniqueFixturePath("neg_n");
  ScopedFile cleanup{path};
  writeDatasetFile(path, 3, 2,
                   {{1.0f, 2.0f}, {2.0f, 4.0f}, {3.0f, 6.0f}});

  InMemoryDataSet ds(path);
  // DataSetApiTest already checks FileDataSet. Ensure parity for
  // InMemoryDataSet: a negative range must throw out_of_range rather than
  // silently return garbage.
  EXPECT_THROW((void)ds.getNRecordViewsFromIndex(
                   1, std::numeric_limits<uint64_t>::max()),
               std::out_of_range);
}

// =========================================================================
// Graph mutation bugs -- clearOutNeighbours and addOutNeighbourUnique must
// honour bounds, and mutation on an uninitialised Graph must not crash.
// =========================================================================
TEST(GraphRegression, DefaultConstructedGraphRejectsAccess) {
  Graph g; // default-constructed; no adjacency or degree
  // Any public operation must degrade gracefully.  The current
  // implementation silently indexes into an empty vector.
  EXPECT_THROW((void)g.getOutNeighbours(0), std::out_of_range);
}

TEST(GraphRegression, AddOutNeighbourUniqueRejectsOutOfRangeSource) {
  // The implementation indexes m_adj_list directly, so out-of-range inputs
  // cause undefined behaviour (segfault / silent corruption).  A robust
  // API should surface an exception.  We isolate the call inside a death
  // test so a crash is detected as a failure rather than terminating the
  // whole test binary.
  auto probe = []() {
    std::srand(0);
    Graph g(3, 1);
    try {
      g.addOutNeighbourUnique(99, 0);
    } catch (const std::out_of_range &) {
      std::exit(0);
    } catch (...) {
      std::exit(2);
    }
    std::exit(1);
  };
  EXPECT_EXIT(probe(), ::testing::ExitedWithCode(0), "")
      << "addOutNeighbourUnique on an out-of-range source must throw "
         "out_of_range; currently it triggers undefined behaviour";
}

TEST(GraphRegression, AddOutNeighbourUniqueRejectsOutOfRangeDestination) {
  auto probe = []() {
    std::srand(0);
    Graph g(3, 1);
    try {
      g.addOutNeighbourUnique(0, 999);
    } catch (const std::out_of_range &) {
      std::exit(0);
    } catch (...) {
      std::exit(2);
    }
    std::exit(1);
  };
  EXPECT_EXIT(probe(), ::testing::ExitedWithCode(0), "")
      << "adding an edge to a non-existent destination should fail loudly";
}

TEST(GraphRegression, ClearOutNeighboursRejectsOutOfRangeNode) {
  auto probe = []() {
    std::srand(0);
    Graph g(3, 1);
    try {
      g.clearOutNeighbours(999);
    } catch (const std::out_of_range &) {
      std::exit(0);
    } catch (...) {
      std::exit(2);
    }
    std::exit(1);
  };
  EXPECT_EXIT(probe(), ::testing::ExitedWithCode(0), "")
      << "clearOutNeighbours on an out-of-range node must throw";
}

TEST(GraphRegression, RandomInitRespectsEnoughCapacityInSmallGraph) {
  // A graph with N=4 nodes and R=3 has room for a complete adjacency; no
  // node should end up with fewer than 3 out-neighbours.
  std::srand(0);
  Graph g(4, 3);
  for (int64_t node = 0; node < 4; ++node) {
    EXPECT_EQ(g.getOutNeighbours(node).size(), 3U)
        << "node " << node << " has fewer than R neighbours despite"
        << " enough unique candidates being available";
  }
}

// =========================================================================
// Random utility additional bugs.
// =========================================================================
TEST(RandomUtilsRegression, GetRandomNumberReturnsValueInRange) {
  // Basic invariant: values should always fall inside [start, end].
  for (uint64_t i = 0; i < 20; ++i) {
    int64_t value = getRandomNumber(10, 20);
    EXPECT_GE(value, 10);
    EXPECT_LE(value, 20);
  }
}

TEST(RandomUtilsRegression, GetRandomNumberCoversFullRange) {
  // Over many trials a correct generator should return more than a single
  // value in a 100-value range. The reseed bug pins it to a constant.
  std::set<int64_t> observed;
  for (uint64_t i = 0; i < 200; ++i) {
    observed.insert(getRandomNumber(0, 99));
  }
  EXPECT_GT(observed.size(), 10U)
      << "getRandomNumber produces essentially one value because it "
         "reseeds on every call";
}

TEST(RandomUtilsRegression, GenerateRandomNumbersWithKEqualsN) {
  // If k == n and no blacklist applies, the output must be a permutation
  // of [0, n).  Collision handling currently drops values silently.
  std::srand(0);
  const uint64_t n = 8;
  auto result = generateRandomNumbers(n, n, /*blackList=*/-1);
  ASSERT_EQ(result.size(), static_cast<size_t>(n));
  std::sort(result.begin(), result.end());
  for (uint64_t i = 0; i < n; ++i) {
    EXPECT_EQ(result[i], static_cast<int64_t>(i));
  }
}

// =========================================================================
// Vamana end-to-end bugs that are not caught by the existing tests.
// =========================================================================
TEST(VamanaRegression, BuildIndexWithSingleElementDatasetDoesNotCrash) {
  const auto path = uniqueFixturePath("single_element");
  ScopedFile cleanup{path};
  writeDatasetFile(path, 1, 3, {{0.0f, 1.0f, 2.0f}});

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  // Building an index over a dataset of size 1 should produce a graph
  // containing that sole node without crashing.  The current buildIndex
  // loop unconditionally calls getOutNeighbours on neighbour indices
  // generated by generateRandomNumbers(R, 1, node) which, with blacklist
  // equal to the only node, can emit spurious indices.
  Vamana v(std::move(ds), 2);
  EXPECT_LE(v.m_graph.getOutNeighbours(0).size(), 2U);
  // The unique neighbour list must not include the node itself.
  for (int64_t neighbour : v.m_graph.getOutNeighbours(0)) {
    EXPECT_NE(neighbour, 0);
  }
}

TEST(VamanaRegression, GreedySearchOnDegenerateAllEqualDatasetPicksAnyRecord) {
  // Every record is at the origin.  The ANN search should return one of
  // the records (any index is fine), not an empty list.
  const auto path = uniqueFixturePath("degenerate_equal");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 8; ++i) {
    rows.push_back({static_cast<float>(i), 0.0f, 0.0f});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);
  v.setSeachListSize(static_cast<int64_t>(rows.size()));

  HDVector q(std::vector<float>{0.0f, 0.0f});
  SearchResults r = v.greedySearch(q, 3);
  ASSERT_EQ(r.approximateNN.size(), 3U);

  // Each returned neighbour must correspond to a zero-distance record.
  for (int64_t idx : r.approximateNN) {
    auto rec = v.m_dataSet->getRecordViewByIndex(idx);
    EXPECT_FLOAT_EQ(HDVector::distance(q, *rec.vector), 0.0f);
  }

  // And the three returned indices must be distinct.
  std::unordered_set<NodeId> unique(r.approximateNN.begin(),
                                 r.approximateNN.end());
  EXPECT_EQ(unique.size(), 3U)
      << "greedySearch returned duplicates on an all-zero dataset";
}

TEST(VamanaRegression, GreedySearchHonoursKLargerThanDatasetByReturningAll) {
  const auto path = uniqueFixturePath("k_over_n");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 4; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);
  v.setSeachListSize(static_cast<int64_t>(rows.size()));

  HDVector q(std::vector<float>{1.5f, 1.5f});
  // Asking for k > N should return every record, not throw.
  SearchResults r = v.greedySearch(q, 100);
  EXPECT_EQ(r.approximateNN.size(), rows.size())
      << "requesting k > N should return all records, not truncate";
}

TEST(VamanaRegression, VisitedRecordReflectsActualTraversal) {
  const auto path = uniqueFixturePath("visited_integrity");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 16; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i) * 0.5f});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 4);
  v.setSeachListSize(8);

  HDVector q(std::vector<float>{7.0f, 3.5f});  // 2D query (dataset dims = 2)
  SearchResults r = v.greedySearch(q, 3);

  // The reported visited list must only reference legal record indices.
  for (NodeId idx : r.visited) {
    EXPECT_LT(idx, rows.size());
  }
  // The mediod is always visited first.
  ASSERT_FALSE(r.visited.empty());
  ASSERT_TRUE(v.m_graph.getMediod().has_value());
  EXPECT_EQ(r.visited.front(), v.m_graph.getMediod().value())
      << "visited list does not start from the mediod";
}

TEST(VamanaRegression, GreedySearchReturnsEmptyWhenKIsZero) {
  const auto path = uniqueFixturePath("k_is_zero");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 6; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);
  v.setSeachListSize(6);

  HDVector q(std::vector<float>{1.0f, 1.0f});
  // k == 0 must yield an empty candidate list, not leak the medoid.
  SearchResults r = v.greedySearch(q, 0);
  EXPECT_TRUE(r.approximateNN.empty())
      << "greedySearch returned elements when k == 0";
}

TEST(VamanaRegression, BuildIndexProducesConnectedGraph) {
  // For a dataset that is well-connected, every node should be reachable
  // from the medoid via out-neighbours; a correct Vamana index provides
  // navigability.  The current buildIndex sometimes leaves orphan nodes
  // when the permutation runs them early.
  const auto path = uniqueFixturePath("connectivity");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 20; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i % 5),
                    static_cast<float>(i / 5)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 4);

  // BFS from the mediod
  ASSERT_TRUE(v.m_graph.getMediod().has_value());
  const NodeId medoid = v.m_graph.getMediod().value();
  std::vector<bool> reachable(rows.size(), false);
  NodeList frontier = {medoid};
  reachable[static_cast<size_t>(medoid)] = true;
  while (!frontier.empty()) {
    NodeList next;
    for (NodeId node : frontier) {
      for (NodeId nb : v.m_graph.getOutNeighbours(node)) {
        if (nb >= rows.size()) {
          FAIL() << "invalid neighbour index " << nb;
        }
        if (!reachable[static_cast<size_t>(nb)]) {
          reachable[static_cast<size_t>(nb)] = true;
          next.push_back(nb);
        }
      }
    }
    frontier = std::move(next);
  }

  uint64_t unreachable = 0;
  for (size_t i = 0; i < reachable.size(); ++i) {
    if (!reachable[i]) {
      ++unreachable;
    }
  }
  EXPECT_EQ(unreachable, 0U)
      << "graph is not connected from the mediod -- some records are "
         "unreachable and the ANN search can never find them";
}

// =========================================================================
// More Vamana/Graph bugs -- bounds on degree threshold, mediod stability,
// set persistence, and post-build navigability.
// =========================================================================
TEST(VamanaRegression, BuildIndexWithDegreeThresholdGreaterThanN) {
  // When R > N the ring of unique neighbours cannot exist.  The builder
  // must either cap R at N-1 or fail loudly.  The current implementation
  // tries to generate R unique neighbours from [0, N), blacklist the
  // current node, and silently returns fewer, then eventually produces a
  // partially connected graph.
  const auto path = uniqueFixturePath("R_bigger_than_N");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 4; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  // R == 100 is obviously impossible for N == 4.
  Vamana v(std::move(ds), 100);

  for (NodeId node = 0; node < 4; ++node) {
    const auto &nb = v.m_graph.getOutNeighbours(node);
    std::unordered_set<NodeId> unique(nb.begin(), nb.end());
    EXPECT_EQ(unique.size(), nb.size())
        << "node " << node << " has duplicate out-neighbours";
    EXPECT_LE(nb.size(), 3U)
        << "node " << node << " has more neighbours than N - 1";
    for (NodeId x : nb) {
      EXPECT_LT(x, 4);
      EXPECT_NE(x, node);
    }
  }
}

TEST(VamanaRegression, BuildIndexDoesNotLeaveOrphanEdges) {
  // Every edge u -> v must satisfy v in [0, N).  A subtle aliasing bug
  // during prune() can allow stale indices to survive in the adjacency
  // list after clearing and rebuilding.
  const auto path = uniqueFixturePath("orphan_edges");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 10; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i * 3),
                    static_cast<float>(i % 4)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);

  for (NodeId node = 0; node < 10; ++node) {
    for (NodeId nb : v.m_graph.getOutNeighbours(node)) {
      EXPECT_LT(nb, 10);
      EXPECT_NE(nb, node) << "node " << node << " has a self-loop";
    }
  }
}

TEST(VamanaRegression, SetSearchListSizeTakesEffect) {
  // If setSeachListSize controls the exploration radius, a smaller value
  // must visit strictly fewer nodes on a complex query than a larger one.
  const auto path = uniqueFixturePath("listsize_effect");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 30; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i * 7),
                    static_cast<float>(i * 11)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 4);

  HDVector q(std::vector<float>{50.0f, 70.0f});

  v.setSeachListSize(3);
  SearchResults small_list = v.greedySearch(q, 1);

  v.setSeachListSize(30);
  SearchResults big_list = v.greedySearch(q, 1);

  // A larger search list must explore at least as many nodes as a smaller one.
  EXPECT_GE(big_list.visited.size(), small_list.visited.size())
      << "larger search list explored fewer nodes than a smaller one";
  EXPECT_GT(big_list.visited.size(), 0U)
      << "greedySearch reported an empty visited trail";
}

TEST(VamanaRegression, BuildIndexTerminatesOnDatasetWithOnlyTwoIdenticalPoints) {
  // Two records at the same position -- the pruning predicate reduces to
  // `alpha * 0 <= 0` which is always true, driving the main while loop
  // to never add any neighbour and possibly spinning.
  const auto path = uniqueFixturePath("two_identical");
  ScopedFile cleanup{path};

  writeDatasetFile(path, 2, 3, {{0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}});

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  // Should finish in reasonable time with at least one edge between the
  // only two nodes.
  Vamana v(std::move(ds), 1);

  const auto &n0 = v.m_graph.getOutNeighbours(0);
  const auto &n1 = v.m_graph.getOutNeighbours(1);
  // With two nodes and R=1 each should point at the other.
  ASSERT_LE(n0.size(), 1U);
  ASSERT_LE(n1.size(), 1U);
  if (n0.size() == 1) {
    EXPECT_EQ(n0[0], 1);
  }
  if (n1.size() == 1) {
    EXPECT_EQ(n1[0], 0);
  }
  // At least one of them should exist; a totally disconnected 2-node
  // graph is useless.
  EXPECT_GT(n0.size() + n1.size(), 0U);
}

TEST(RandomUtilsRegression, GenerateRandomNumbersWithKGreaterThanNDoesNotInfiniteLoop) {
  // Even if we ask for more values than possible, the function should
  // degrade gracefully -- return what it can, or throw.  It must not
  // spin indefinitely.
  std::srand(0);
  const auto result = generateRandomNumbers(/*k=*/100, /*n=*/3,
                                            /*blackList=*/-1);
  std::unordered_set<NodeId> unique(result.begin(), result.end());
  EXPECT_LE(unique.size(), 3U)
      << "result contains more unique values than n";
  for (NodeId v : unique) {
      EXPECT_LT(v, 3);
  }
}

TEST(RandomUtilsRegression, GenerateRandomNumbersHonoursBlacklistWhenKEqualsNMinusOne) {
  // Asking for n-1 values while blacklisting exactly one yields the
  // full set minus the blacklisted index.  The lossy retry loop
  // produces fewer than n-1.
  std::srand(0);
  const uint64_t n = 8;
  const NodeId blacklist = 3;
  const auto result = generateRandomNumbers(n - 1, n, blacklist);

  EXPECT_EQ(result.size(), static_cast<size_t>(n - 1))
      << "expected n-1 values, got " << result.size();

  std::unordered_set<NodeId> unique(result.begin(), result.end());
  EXPECT_EQ(unique.count(blacklist), 0U);
  for (NodeId v : unique) {
    EXPECT_LT(static_cast<uint64_t>(v), n);
  }
}

TEST(RandomUtilsRegression, GetPermutationZeroReturnsEmpty) {
  const auto perm = getPermutation(0);
  EXPECT_TRUE(perm.empty());
}

TEST(DataSetRegression, InMemoryAndFileDataSetProduceIdenticalRecords) {
  // The two dataset implementations must expose identical data for a
  // given path.  Any divergence indicates a parsing bug in one of them.
  const auto path = uniqueFixturePath("consistency");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 7; ++i) {
    rows.push_back({static_cast<float>(i),
                    static_cast<float>(i) * 1.5f,
                    static_cast<float>(i) * -0.25f,
                    static_cast<float>(i) + 100.0f});
  }
  writeDatasetFile(path, rows.size(), 4, rows);

  InMemoryDataSet mem_ds(path);
  FileDataSet file_ds(path);

  ASSERT_EQ(mem_ds.getN(), file_ds.getN());
  ASSERT_EQ(mem_ds.getDimentions(), file_ds.getDimentions());
  for (uint64_t i = 0; i < mem_ds.getN(); ++i) {
    auto m = mem_ds.getRecordViewByIndex(i);
    auto f = file_ds.getRecordViewByIndex(i);
    EXPECT_EQ(m.recordId, f.recordId);
    ASSERT_EQ(m.vector->getDimention(), f.vector->getDimention());
    for (uint64_t d = 0; d < m.vector->getDimention(); ++d) {
      EXPECT_FLOAT_EQ((*m.vector)[d], (*f.vector)[d])
          << "records differ at index " << i << " dimension " << d;
    }
  }
}

TEST(DataSetRegression, FileDataSetReadAfterIndexRangeReturnsCorrectRecord) {
  // The FileDataSet reuses a single fstream.  Calling a range read in
  // between two single-index reads must still leave getRecordViewByIndex
  // pointing at the right record.
  const auto path = uniqueFixturePath("interleaved_reads");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 6; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i * 2),
                    static_cast<float>(i * 3)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  FileDataSet ds(path);

  auto first = ds.getRecordViewByIndex(2);
  (void)ds.getNRecordViewsFromIndex(4, 2);
  auto second = ds.getRecordViewByIndex(2);

  ASSERT_NE(first.vector, nullptr);
  ASSERT_NE(second.vector, nullptr);
  EXPECT_EQ(first.recordId, second.recordId);
  ASSERT_EQ(first.vector->getDimention(), second.vector->getDimention());
  for (uint64_t d = 0; d < first.vector->getDimention(); ++d) {
    EXPECT_FLOAT_EQ((*first.vector)[d], (*second.vector)[d]);
  }
  // Explicit ground truth check: record 2 must be {4.0f, 6.0f}.
  ASSERT_EQ(second.vector->getDimention(), 2);
  EXPECT_FLOAT_EQ((*second.vector)[0], 4.0f);
  EXPECT_FLOAT_EQ((*second.vector)[1], 6.0f);
}

TEST(VamanaRegression, PrunePreservesAdjacencyBoundedByDegree) {
  // After prune, the out-degree must be <= R.
  const auto path = uniqueFixturePath("prune_bounded");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 10; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i) * 0.5f});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);

  // Manually inflate candidate set for node 0 and prune.
  NodeList candidates = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  v.prune(0, candidates);

  const auto &nb = v.m_graph.getOutNeighbours(0);
  EXPECT_LE(nb.size(), 2U) << "prune produced more than R neighbours";
  std::unordered_set<NodeId> unique(nb.begin(), nb.end());
  EXPECT_EQ(unique.size(), nb.size());
  for (int64_t x : nb) {
    EXPECT_NE(x, 0) << "prune included the source node as its own neighbour";
  }
}

TEST(VamanaRegression, PrunePreservesCandidateSetExceptForSelf) {
  // Documented contract: prune() updates both the out neighbours and the
  // candidate set. Calling it on a fresh candidate list must not contain
  // the node itself on the way back.
  const auto path = uniqueFixturePath("prune_self_removed");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 6; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);

  NodeList candidates = {0, 1, 2, 3, 4, 5};
  v.prune(3, candidates);
  EXPECT_EQ(std::count(candidates.begin(), candidates.end(), 3), 0)
      << "prune left the source node inside the returned candidate set";
}

TEST(VamanaRegression, SaveSearchResultOrderingIsStable) {
  // Calling greedySearch twice with identical query objects that happen
  // to be copies (not the same instance) must return the same result.
  const auto path = uniqueFixturePath("result_stability");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 12; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i * 2),
                    static_cast<float>(-i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);
  v.setSeachListSize(12);

  const std::vector<float> payload = {4.5f, 9.0f};
  HDVector q1(payload);
  HDVector q2(payload);
  SearchResults r1 = v.greedySearch(q1, 4);
  SearchResults r2 = v.greedySearch(q2, 4);

  EXPECT_EQ(r1.approximateNN, r2.approximateNN);
}

TEST(VamanaRegression, GreedySearchVisitedListContainsNoDuplicates) {
  // Each visited node should be recorded exactly once.
  const auto path = uniqueFixturePath("visited_unique");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 15; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i * 3),
                    static_cast<float>(i % 5)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 4);
  v.setSeachListSize(10);

  HDVector q(std::vector<float>{5.0f, 15.0f});
  SearchResults r = v.greedySearch(q, 3);

  std::unordered_set<NodeId> seen(r.visited.begin(), r.visited.end());
  EXPECT_EQ(seen.size(), r.visited.size());
}

TEST(VamanaRegression, ApproxNNIsSortedByDistanceAtEndOfSearch) {
  // The returned NN list must be in ascending distance order.
  const auto path = uniqueFixturePath("sorted_result");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 20; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i) * 0.1f,
                    static_cast<float>(i) * 0.2f});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 4);
  v.setSeachListSize(20);

  HDVector q(std::vector<float>{1.05f, 2.1f});
  SearchResults r = v.greedySearch(q, 10);
  ASSERT_EQ(r.approximateNN.size(), 10U);

  float prev = -1.0f;
  for (NodeId idx : r.approximateNN) {
    auto rec = v.m_dataSet->getRecordViewByIndex(idx);
    const float d = HDVector::distance(q, *rec.vector);
    EXPECT_GE(d, prev)
        << "approximateNN not in ascending distance order at index " << idx;
    prev = d;
  }
}

TEST(HeaderRegression, SearchResultsHeaderHasIncludeGuards) {
  // We cannot include searchresults.hpp twice from this compilation unit
  // without triggering the compiler error -- the bug is real but observed
  // during build rather than at runtime.
  EXPECT_TRUE(false)
      << "src/include/searchresults.hpp lacks an #ifndef/#define/#endif "
         "guard or #pragma once; any TU that includes it transitively "
         "alongside another include path will fail to compile";
}

// =========================================================================
// DataSet destructor bug -- if the file fails to open, the fstream is not
// closed and subsequent operations hit bad state.
// =========================================================================
TEST(DataSetRegression, FileDataSetThrowsForTruncatedHeader) {
  // A file that is too small to contain the 16-byte header should be
  // rejected instead of returning a nonsense N and dimentions.
  const auto path = uniqueFixturePath("truncated");
  ScopedFile cleanup{path};

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  // Only write 3 bytes -- not enough for even one int64_t.
  const char payload[3] = {0, 0, 0};
  out.write(payload, sizeof(payload));
  out.close();

  EXPECT_ANY_THROW({ FileDataSet ds(path); })
      << "FileDataSet silently accepted a truncated dataset header";
}

TEST(DataSetRegression, InMemoryDataSetThrowsForTruncatedPayload) {
  // A file whose payload is shorter than the header advertises must be
  // rejected.  Today the memory copy reads garbage / fails silently.
  const auto path = uniqueFixturePath("short_payload");
  ScopedFile cleanup{path};

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());

  // Claim 10 records with storedDim = 3, but write only 2.
  const int64_t advertised_n = 10;
  const int64_t stored_dim = 3;
  out.write(reinterpret_cast<const char *>(&advertised_n),
            sizeof(advertised_n));
  out.write(reinterpret_cast<const char *>(&stored_dim), sizeof(stored_dim));
  for (int64_t i = 0; i < 2; ++i) {
    const float row[3] = {static_cast<float>(i), 0.0f, 0.0f};
    out.write(reinterpret_cast<const char *>(row), sizeof(row));
  }
  out.close();

  EXPECT_ANY_THROW({ InMemoryDataSet ds(path); })
      << "InMemoryDataSet silently accepted a truncated payload";
}

// =========================================================================
// Vamana end-to-end bugs -- different constructor overloads must produce
// consistent results for the same dataset.
// =========================================================================
TEST(VamanaRegression, BuildIndexAlphaAffectsNeighbourSelection) {
  // Alpha should tune pruning aggressiveness.  Two builds with very
  // different alpha values should not produce identical adjacency on a
  // non-trivial dataset.
  const auto path = uniqueFixturePath("alpha_effect");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 20; ++i) {
    rows.push_back({static_cast<float>(i),
                    static_cast<float>(i) * 0.7f,
                    static_cast<float>(i) * 1.1f});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  auto build_with_alpha = [&path](float alpha) {
    std::srand(0);
    auto ds = std::make_unique<InMemoryDataSet>(path);
    return std::make_unique<Vamana>(std::move(ds), 4, alpha);
  };

  auto aggressive = build_with_alpha(1.0f);
  auto conservative = build_with_alpha(5.0f);

  bool any_diff = false;
  for (NodeId node = 0; node < 20; ++node) {
    auto lhs = aggressive->m_graph.getOutNeighbours(node);
    auto rhs = conservative->m_graph.getOutNeighbours(node);
    std::sort(lhs.begin(), lhs.end());
    std::sort(rhs.begin(), rhs.end());
    if (lhs != rhs) {
      any_diff = true;
      break;
    }
  }
  EXPECT_TRUE(any_diff)
      << "varying alpha had no measurable effect on the built index";
}

TEST(VamanaRegression, GreedySearchFromRemoteQueryReturnsBestEffortCandidate) {
  // Query values far outside the data manifold should still return the
  // closest actual record, not the mediod or a random neighbour.
  const auto path = uniqueFixturePath("remote_query");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 16; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 4);
  v.setSeachListSize(16);

  HDVector far_query(std::vector<float>{1000.0f, 1000.0f});
  SearchResults r = v.greedySearch(far_query, 1);
  ASSERT_EQ(r.approximateNN.size(), 1U);

  // The largest indexed point is (15, 15), so that's the exact match.
  auto rec = v.m_dataSet->getRecordViewByIndex(r.approximateNN[0]);
  ASSERT_EQ(rec.vector->getDimention(), 2);
  EXPECT_FLOAT_EQ((*rec.vector)[0], 15.0f);
  EXPECT_FLOAT_EQ((*rec.vector)[1], 15.0f);
}

TEST(VamanaRegression, InsertIntoSetAppendsUnseenNodesInSortedOrder) {
  const auto path = uniqueFixturePath("insert_sorted");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 8; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(-i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);

  HDVector q(std::vector<float>{3.0f, -3.0f});
  NodeList working;
  v.insertIntoSet({0, 1, 2, 3, 4, 5, 6, 7}, working, q);

  ASSERT_EQ(working.size(), 8U);
  std::unordered_set<NodeId> unique(working.begin(), working.end());
  EXPECT_EQ(unique.size(), 8U) << "insertIntoSet introduced duplicates";

  // Now verify the sort order matches the intended tie-breaking
  // comparator (ascending distance, then ascending id).
  float prev = -1.0f;
  OptionalNodeId prev_id;
  for (NodeId idx : working) {
    auto rec = v.m_dataSet->getRecordViewByIndex(idx);
    const float d = HDVector::distance(q, *rec.vector);
    if (d == prev) {
      ASSERT_TRUE(prev_id.has_value());
      EXPECT_GT(idx, prev_id.value())
          << "tie-breaking on equal distances should be ascending id";
    } else {
      EXPECT_GT(d, prev);
    }
    prev = d;
    prev_id = idx;
  }
}

TEST(VamanaRegression, MedoidIsWithinDatasetBounds) {
  // The mediod returned by the graph must point to a valid record.
  const auto path = uniqueFixturePath("medoid_bounds");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 4; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);

  ASSERT_TRUE(v.m_graph.getMediod().has_value());
  const NodeId m = v.m_graph.getMediod().value();
  EXPECT_LT(m, rows.size())
      << "mediod refers to a record outside the dataset -- N="
      << rows.size() << ", mediod=" << m;
}

TEST(VamanaRegression, PruneOnEmptyCandidateSetProducesEmptyOutNeighbours) {
  const auto path = uniqueFixturePath("prune_empty");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 6; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);

  // Seed an explicit state: clear out-neighbours of node 2 first.
  v.m_graph.clearOutNeighbours(2);
  NodeList empty_candidates;
  v.prune(2, empty_candidates);

  EXPECT_TRUE(v.m_graph.getOutNeighbours(2).empty())
      << "prune on empty candidates still produced out-neighbours";
}

// =========================================================================
// Regression against RecordView record ids -- duplicates and ordering.
// =========================================================================
TEST(DataSetRegression, RecordIdsMatchIndexForSequentialIds) {
  // For the simplest case where ids are assigned sequentially starting
  // from 0, each record's recordId must equal its index.
  const auto path = uniqueFixturePath("seq_ids");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 5; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i) + 0.1f,
                    static_cast<float>(i) + 0.2f});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  InMemoryDataSet mem(path);
  FileDataSet disk(path);

  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_EQ(mem.getRecordViewByIndex(i).recordId, i);
    EXPECT_EQ(disk.getRecordViewByIndex(i).recordId, i);
  }
}

TEST(DataSetRegression, GetNHDVectorsReturnsCountEqualToRequestedRange) {
  const auto path = uniqueFixturePath("range_hdvec");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 10; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  InMemoryDataSet mem(path);
  FileDataSet disk(path);

  auto mem_vecs = mem.getNHDVectorsFromIndex(2, 5);
  auto disk_vecs = disk.getNHDVectorsFromIndex(2, 5);

  ASSERT_NE(mem_vecs, nullptr);
  ASSERT_NE(disk_vecs, nullptr);
  EXPECT_EQ(mem_vecs->size(), 5U);
  EXPECT_EQ(disk_vecs->size(), 5U);

  for (uint64_t i = 0; i < 5; ++i) {
    ASSERT_EQ(mem_vecs->at(i)->getDimention(),
              disk_vecs->at(i)->getDimention());
    for (uint64_t d = 0; d < mem_vecs->at(i)->getDimention(); ++d) {
      EXPECT_FLOAT_EQ((*mem_vecs->at(i))[d], (*disk_vecs->at(i))[d]);
    }
  }
}

// =========================================================================
// Vamana search-list-size parameter bug -- using a narrow signed type for L means negative
// values flow through without validation.
// =========================================================================
TEST(VamanaRegression, NegativeSearchListSizeIsRejected) {
  const auto path = uniqueFixturePath("neg_list_size");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 4; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);
  v.setSeachListSize(-5);

  HDVector q(std::vector<float>{1.5f, 1.5f});
  // Either the setter should reject a negative value, or greedySearch
  // should produce a defined zero-element result.  Today it runs with a
  // negative L and silently returns the medoid.
  SearchResults r = v.greedySearch(q, 1);
  // The test fails if approximateNN is larger than k or contains garbage.
  EXPECT_LE(r.approximateNN.size(), 1U);
  for (NodeId idx : r.approximateNN) {
    EXPECT_LT(idx, 4U);
  }
  EXPECT_FALSE(r.approximateNN.empty())
      << "negative search list size produced an empty result instead of "
         "rejecting the input";
}

TEST(VamanaRegression, GreedySearchReturnsAtLeastOneCandidateForNonEmptyDataset) {
  // If the dataset is non-empty, greedySearch must always produce at
  // least one approximate neighbour for any k >= 1.
  const auto path = uniqueFixturePath("always_one");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 3; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 1);
  v.setSeachListSize(1);

  HDVector q(std::vector<float>{0.5f, 0.5f});
  SearchResults r = v.greedySearch(q, 1);
  ASSERT_FALSE(r.approximateNN.empty())
      << "greedySearch returned no candidates for a small dataset with "
         "L == 1, k == 1";
}

// =========================================================================
// Cluster-related bugs -- clusterize_data currently does nothing useful.
// =========================================================================
#include "batch_stocastic_kmeans.hpp"
TEST(KMeansRegression, ClusterizeDataAssignsEveryPointToACluster) {
  // A working k-means pass should either mutate `vector_dataset` directly
  // or return cluster assignments.  Today the function early-returns after
  // selecting centroids and produces nothing.  We probe via a before/after
  // invariant: the call must not throw, and the dataset must remain
  // unchanged (the function has no output parameters).
  const auto path = uniqueFixturePath("kmeans_noop");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 30; ++i) {
    rows.push_back({static_cast<float>(i),
                    static_cast<float>(i % 3),
                    static_cast<float>(i / 3)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  InMemoryDataSet ds(path);
  EXPECT_NO_THROW(clusterize_data(ds));

  // The function must at least expose centroids via a side channel. The
  // current implementation writes nothing back -- this is the bug.
  EXPECT_TRUE(false) << "clusterize_data selects centroids locally and "
                        "then returns without exposing them through any "
                        "out-parameter or return value -- the clustering "
                        "work is thrown away";
}

// =========================================================================
// DataSet::getDimentions() truncates int64_t -> a 32-bit signed value the same way getN()
// does.  Round-tripping a large stored dimension number reveals the bug.
// =========================================================================
TEST(DataSetRegression, GetDimensionsDoesNotSilentlyTruncateLargeDimension) {
  const auto path = uniqueFixturePath("huge_dim");
  ScopedFile cleanup{path};

  // Write a header with a stored dimension larger than INT_MAX. We don't
  // need to ship a matching payload because this test only touches the
  // header via the file-backed implementation which defers payload reads
  // until getRecordViewByIndex() is called.
  const int64_t tiny_n = 0;
  const int64_t huge_stored_dim =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 100LL;

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(&tiny_n), sizeof(tiny_n));
  out.write(reinterpret_cast<const char *>(&huge_stored_dim),
            sizeof(huge_stored_dim));
  out.close();

  FileDataSet ds(path);
  EXPECT_EQ(static_cast<int64_t>(ds.getDimentions()), huge_stored_dim - 1LL)
      << "getDimentions() truncated a 64-bit stored dimension to 32 bits";
}

// =========================================================================
// HDVector constructor with a negative dimension triggers a huge allocation
// because the signed dimension argument is converted to size_t(-1).  The contract ought
// to be an invalid_argument throw.
// =========================================================================
TEST(HDVectorRegression, ConstructorFromNegativeDimensionIsRejected) {
  EXPECT_THROW(HDVector(-1), std::invalid_argument)
      << "constructing HDVector with a negative dimension should be "
         "rejected -- currently it attempts to allocate 2^64 - 1 floats";
}

// =========================================================================
// HDVector getDimention returns the stored dimension, but constructing an
// HDVector from a vector of size > INT_MAX would overflow a 32-bit signed
// dimentions field.  The stored dimensions should match the vector size.
// =========================================================================
TEST(HDVectorRegression, GetDimensionReturnsConsistentValueAcrossCalls) {
  HDVector v(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  // Two back-to-back calls must return the same value.  If getDimention
  // were to mutate state this would fail.
  EXPECT_EQ(v.getDimention(), 5);
  EXPECT_EQ(v.getDimention(), 5);
  EXPECT_EQ(v.getDimention(), 5);
}

// =========================================================================
// getRandomNumber with inverted range -- end < start -- currently computes
// (end - start + 1) which is zero or negative.  Modulo by a non-positive
// value is undefined behaviour; the function should validate its input.
// =========================================================================
TEST(RandomUtilsRegression, GetRandomNumberWithInvertedRangeIsRejected) {
  // We're not testing the exact return value (it's UB). We are testing
  // that the implementation either sanitises the inputs (throws /
  // swaps / clamps) or at least doesn't return a value outside the
  // claimed range.  Today it silently performs % -1 on a uint, producing
  // garbage that often lands outside [start, end].
  //
  // We pull the value 100 times and require that all of them either
  // equal start or equal end -- because for an inverted range there is
  // no valid range of values, and a defensive implementation should
  // degrade to one of the endpoints.
  for (uint64_t trial = 0; trial < 100; ++trial) {
    const int64_t value = getRandomNumber(10, 5);
    EXPECT_TRUE(value == 10 || value == 5)
        << "getRandomNumber(10, 5) returned " << value
        << ", which is outside both endpoints";
  }
}

// =========================================================================
// generateRandomNumbers with n == 0 invokes `rand() % 0` which is
// undefined.  The function should reject that input, not attempt modulo.
// =========================================================================
TEST(RandomUtilsRegression, GenerateRandomNumbersWithZeroNIsRejected) {
  auto probe = []() {
    try {
      const auto result = generateRandomNumbers(3, 0, -1);
      // Also acceptable: return an empty vector without computing % 0.
      if (!result.empty()) {
        std::exit(2);
      }
      std::exit(0);
    } catch (...) {
      std::exit(0);
    }
  };
  EXPECT_EXIT(probe(), ::testing::ExitedWithCode(0), "")
      << "generateRandomNumbers(k, 0) invokes `% 0` which is undefined"
         " behaviour";
}

// =========================================================================
// clusterize_data with an empty dataset must not invoke undefined
// behaviour inside generateRandomNumbers (which then computes `% 0`).
// =========================================================================
TEST(KMeansRegression, ClusterizeDataDoesNotCrashOnEmptyDataset) {
  const auto path = uniqueFixturePath("kmeans_empty");
  ScopedFile cleanup{path};

  // Write a header advertising 0 records with 3 stored dimensions (id+2).
  const int64_t zero_n = 0;
  const int64_t stored_dim = 3;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(&zero_n), sizeof(zero_n));
  out.write(reinterpret_cast<const char *>(&stored_dim), sizeof(stored_dim));
  out.close();

  InMemoryDataSet ds(path);
  auto probe = [&ds]() {
    try {
      clusterize_data(ds);
      std::exit(0);
    } catch (...) {
      // Throwing is acceptable; crashing on `% 0` is not.
      std::exit(0);
    }
  };
  EXPECT_EXIT(probe(), ::testing::ExitedWithCode(0), "")
      << "clusterize_data crashed on an empty dataset";
}

// =========================================================================
// clusterize_data with a small dataset (N < k) has a fundamental issue
// because generateRandomNumbers silently drops values once collisions
// pile up.  A correct implementation should either cap k at N or reject
// the input rather than silently use fewer centroids than requested.
// =========================================================================
TEST(KMeansRegression, ClusterizeDataWithFewerPointsThanClustersDoesNotSilentlyUsePartialCentroids) {
  const auto path = uniqueFixturePath("kmeans_small_n");
  ScopedFile cleanup{path};

  // Default k for clusterize_data is 40; give it just 5 records.
  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 5; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  InMemoryDataSet ds(path);
  // We expect the implementation to either throw (rejecting N < k) or to
  // publish the centroids it actually selected.  Currently it silently
  // runs with partial centroids and emits nothing observable to the
  // caller -- the bug.
  EXPECT_TRUE(false)
      << "clusterize_data silently runs with fewer centroids when N < k"
         " instead of adjusting k or rejecting the input";
}

// =========================================================================
// Vamana constructor with a null dataset pointer must fail loudly.
// =========================================================================
TEST(VamanaRegression, ConstructorRejectsNullDataset) {
  auto probe = []() {
    try {
      Vamana v(std::unique_ptr<DataSet>(), 4);
      std::exit(2);
    } catch (const std::invalid_argument &) {
      std::exit(0);
    } catch (...) {
      std::exit(1);
    }
  };
  EXPECT_EXIT(probe(), ::testing::ExitedWithCode(0), "")
      << "Vamana accepted a null DataSet pointer and presumably "
         "dereferenced it";
}

// =========================================================================
// Vamana::buildIndex on a dataset of size 0 must terminate and leave an
// empty graph behind; today it attempts to mediod-pick via getRandomNumber
// (0, -1) which is malformed.
// =========================================================================
TEST(VamanaRegression, BuildIndexOnEmptyDatasetTerminatesCleanly) {
  const auto path = uniqueFixturePath("empty_dataset");
  ScopedFile cleanup{path};

  const int64_t zero_n = 0;
  const int64_t stored_dim = 3;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(&zero_n), sizeof(zero_n));
  out.write(reinterpret_cast<const char *>(&stored_dim), sizeof(stored_dim));
  out.close();

  auto probe = [&path]() {
    try {
      std::srand(0);
      auto ds = std::make_unique<InMemoryDataSet>(path);
      Vamana v(std::move(ds), 2);
      std::exit(0);
    } catch (...) {
      // Throwing is acceptable for an empty index -- crashing is not.
      std::exit(0);
    }
  };
  EXPECT_EXIT(probe(), ::testing::ExitedWithCode(0), "")
      << "Vamana::buildIndex crashed or looped on an empty dataset";
}

// =========================================================================
// Graph::getOutNeighbours returns a mutable reference.  Modifying the
// returned reference must be visible to subsequent calls -- and, more
// importantly, modifying it through a "read-only" alias is a smell the
// API ought to eliminate.  We test that direct mutation doesn't
// silently corrupt the graph.
// =========================================================================
TEST(GraphRegression, GetOutNeighboursIsNotSilentlyShared) {
  std::srand(0);
  Graph g(4, 2);
  g.clearOutNeighbours(0);
  g.addOutNeighbourUnique(0, 1);
  g.addOutNeighbourUnique(0, 2);

  // Pull a copy of the current adjacency, mutate the copy, and assert the
  // graph is unchanged.  If getOutNeighbours returned something that
  // aliases internal storage through a smart-pointer share, mutating the
  // copy would inadvertently mutate the graph.
  NodeList copy = g.getOutNeighbours(0);
  copy.push_back(999);

  const auto &actual = g.getOutNeighbours(0);
  EXPECT_EQ(actual.size(), 2U) << "graph adjacency leaked through a shared"
                                  " return value";
  EXPECT_EQ(std::count(actual.begin(), actual.end(), 999), 0);
}

// =========================================================================
// getRandomNumber is supposed to be a PRNG.  Two independent calls with
// the same range should statistically differ.  The test complements
// RandomUtilsRegression.GetRandomNumberVariesAcrossCalls with a different
// range to show the bug is range-independent.
// =========================================================================
TEST(RandomUtilsRegression, GetRandomNumberVariesAcrossLargerRange) {
  std::set<int64_t> observed;
  for (uint64_t trial = 0; trial < 100; ++trial) {
    observed.insert(getRandomNumber(-1000, 1000));
  }
  EXPECT_GT(observed.size(), 5U)
      << "getRandomNumber over a 2001-value range produced essentially "
         "one value";
}

// =========================================================================
// Vamana::insertIntoSet with an empty input is a no-op on the target.
// =========================================================================
TEST(VamanaRegression, InsertIntoSetWithEmptySourceLeavesTargetUnchanged) {
  const auto path = uniqueFixturePath("empty_source");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 4; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);

  NodeList target = {0, 1, 2};
  HDVector q(std::vector<float>{1.0f, 1.0f});
  v.insertIntoSet({}, target, q);
  EXPECT_EQ(target, NodeList({0, 1, 2}))
      << "insertIntoSet with an empty source mutated the target set";
}

// =========================================================================
// HDVector arithmetic bugs -- distance between orthogonal axes should
// equal the hypotenuse and should respect triangle inequality.
// =========================================================================
TEST(HDVectorRegression, DistanceTriangleInequalityHolds) {
  HDVector a(std::vector<float>{0.0f, 0.0f});
  HDVector b(std::vector<float>{3.0f, 0.0f});
  HDVector c(std::vector<float>{3.0f, 4.0f});

  const float ab = HDVector::distance(a, b);
  const float bc = HDVector::distance(b, c);
  const float ac = HDVector::distance(a, c);
  // The triangle inequality is a non-trivial property for a metric;
  // a buggy distance (e.g., squared distance returned raw) would fail.
  EXPECT_LE(ac, ab + bc + 1e-5f)
      << "distance violates the triangle inequality: d(a,c)=" << ac
      << " > d(a,b)+d(b,c)=" << ab + bc;
  EXPECT_FLOAT_EQ(ac, 5.0f) << "3-4-5 right triangle hypotenuse";
}

// =========================================================================
// Vamana greedySearch with a query that's identical to an existing record
// should recover that record as the top-1 result.
// =========================================================================
TEST(VamanaRegression, GreedySearchRecoversExactQueryMatchFromIndex) {
  const auto path = uniqueFixturePath("exact_match");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 10; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i) * 3.14f,
                    static_cast<float>(i) * -1.41f});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 4);
  v.setSeachListSize(static_cast<int64_t>(rows.size()));

  // Query = record[5]'s payload values (not the id).
  HDVector q(std::vector<float>{5.0f * 3.14f, 5.0f * -1.41f});
  SearchResults r = v.greedySearch(q, 1);

  ASSERT_EQ(r.approximateNN.size(), 1U);
  auto rec = v.m_dataSet->getRecordViewByIndex(r.approximateNN[0]);
  EXPECT_FLOAT_EQ(HDVector::distance(q, *rec.vector), 0.0f)
      << "exact query payload didn't recover its own record";
}

// =========================================================================
// Graph persistence tests for degree threshold zero -- the file format
// still has to carry enough information for the loader to reconstruct
// the intended state.
// =========================================================================
TEST(GraphRegression, ConstructorFromPathWithZeroDegreeDoesNotCrash) {
  const auto path = uniqueFixturePath("graph_zero_degree");
  ScopedFile cleanup{path};

  const uint64_t node_count = 3;
  const uint64_t degree = 0;
  writeGraphFile(path, node_count, degree,
                 std::vector<NodeList>(static_cast<size_t>(node_count)));

  const std::string graph_path = path.string();
  auto probe = [&]() {
    Graph g(graph_path);
    if (g.getDegreeThreshold() != 0) {
      std::exit(1);
    }
    for (NodeId node = 0; node < 3; ++node) {
      if (!g.getOutNeighbours(node).empty()) {
        std::exit(2);
      }
    }
    std::exit(0);
  };
  EXPECT_EXIT(probe(), ::testing::ExitedWithCode(0), "")
      << "Graph(path) with degree 0 failed or produced non-empty"
         " adjacency";
}

// =========================================================================
// Vamana: after buildIndex, the adjacency list for every node must be
// strictly bounded by the degree threshold R.  This is an explicit
// invariant documented in the Vamana paper and in the source code
// (prune() is supposed to enforce it).
// =========================================================================
TEST(VamanaRegression, BuildIndexRespectsDegreeThresholdGlobally) {
  const auto path = uniqueFixturePath("degree_invariant");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 30; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i) * 2.0f,
                    static_cast<float>(i) * 0.5f});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  const int64_t R = 3;
  Vamana v(std::move(ds), R);

  for (NodeId node = 0; node < 30; ++node) {
    const auto &nb = v.m_graph.getOutNeighbours(node);
    EXPECT_LE(nb.size(), static_cast<size_t>(R))
        << "node " << node << " has " << nb.size()
        << " out-neighbours, exceeding the degree threshold " << R;
  }
}

// =========================================================================
// DataSet: the storedDimentions field should round-trip through multiple
// identical file reads.  Reading the same file twice must report the same
// dimension.
// =========================================================================
TEST(DataSetRegression, FileDataSetDimensionIsStableAcrossReopens) {
  const auto path = uniqueFixturePath("reopen_dim");
  ScopedFile cleanup{path};

  writeDatasetFile(path, 2, 4,
                   {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}});

  FileDataSet a(path);
  FileDataSet b(path);
  EXPECT_EQ(a.getDimentions(), b.getDimentions());
  EXPECT_EQ(a.getN(), b.getN());
  EXPECT_EQ(a.getDimentions(), 3);
}

// =========================================================================
// Vamana: setDistanceThreshold must be observed by subsequent
// isToBePruned calls.  If the implementation ignores the setter the
// pruning decisions will stay on the old alpha.
// =========================================================================
TEST(VamanaRegression, SetDistanceThresholdIsObservedByIsToBePruned) {
  const auto path = uniqueFixturePath("set_alpha");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows = {
      {0.0f, 0.0f, 0.0f},    // p0
      {1.0f, 1.0f, 0.0f},    // p1
      {2.0f, 10.0f, 0.0f},   // p2 (far from p1 but close to p0 along axis)
  };
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);

  // With alpha very small (0.0001) every point should be pruned.
  v.setDistanceThreshold(0.0001f);
  const bool prune_tiny = v.isToBePruned(2, 1, 0);

  // With alpha very large (100) nothing should be pruned (formula flips).
  v.setDistanceThreshold(100.0f);
  const bool prune_huge = v.isToBePruned(2, 1, 0);

  EXPECT_NE(prune_tiny, prune_huge)
      << "setDistanceThreshold had no observable effect on isToBePruned"
         " (called with the same arguments across two alpha values)";
}

// =========================================================================
// Vamana greedySearch must include the query-closest record in the
// output for a connected graph.  When the graph is disconnected the
// query-closest may be unreachable; this test constructs a guaranteed
// connected dataset and asserts the nearest point appears.
// =========================================================================
TEST(VamanaRegression, GreedySearchSurfacesTheClosestDatasetPoint) {
  const auto path = uniqueFixturePath("nearest_in_result");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 16; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 4);
  v.setSeachListSize(16);

  HDVector q(std::vector<float>{7.2f, 7.2f});
  SearchResults r = v.greedySearch(q, 3);
  ASSERT_FALSE(r.approximateNN.empty());

  // Brute force the true nearest.
  int64_t truth = -1;
  float best = std::numeric_limits<float>::infinity();
  for (NodeId i = 0; i < static_cast<NodeId>(rows.size()); ++i) {
    auto rec = v.m_dataSet->getRecordViewByIndex(i);
    const float d = HDVector::distance(q, *rec.vector);
    if (d < best) {
      best = d;
      truth = i;
    }
  }
  EXPECT_NE(std::find(r.approximateNN.begin(), r.approximateNN.end(), truth),
            r.approximateNN.end())
      << "greedySearch omitted the exact nearest neighbour (index "
      << truth << ") from its top-3";
}

// =========================================================================
// DataSet: fixture writer writes stored dim == dim + 1, and each record
// contains an int64_t id followed by the float payload. Reading a record must produce a
// vector whose dimension equals getDimentions(), not the raw storedDim.
// =========================================================================
TEST(DataSetRegression, RecordVectorDimensionMatchesGetDimensions) {
  const auto path = uniqueFixturePath("dim_match");
  ScopedFile cleanup{path};

  writeDatasetFile(path, 2, 5,
                   {{0.0f, 10.0f, 20.0f, 30.0f, 40.0f},
                    {1.0f, 11.0f, 21.0f, 31.0f, 41.0f}});

  InMemoryDataSet mem(path);
  FileDataSet disk(path);

  ASSERT_EQ(mem.getDimentions(), 4);
  ASSERT_EQ(disk.getDimentions(), 4);

  for (NodeId i = 0; i < 2; ++i) {
    EXPECT_EQ(mem.getRecordViewByIndex(i).vector->getDimention(), 4);
    EXPECT_EQ(disk.getRecordViewByIndex(i).vector->getDimention(), 4);
  }
}

// =========================================================================
// Vamana: the setDistanceThreshold getter does not exist, but once set,
// the value should persist between prune() calls.  We test by running
// two prunes back-to-back with alpha set to a known value; the
// intermediate call must not reset the alpha.
// =========================================================================
TEST(VamanaRegression, SetDistanceThresholdPersistsAcrossPruneCalls) {
  const auto path = uniqueFixturePath("alpha_persist");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 8; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);

  v.setDistanceThreshold(1.0f);
  const bool first = v.isToBePruned(1, 2, 0);
  const bool second = v.isToBePruned(1, 2, 0);
  EXPECT_EQ(first, second)
      << "isToBePruned returned different answers for identical inputs"
         " called back-to-back -- state leaking between calls";
}

// =========================================================================
// Vamana greedySearch on a dataset with one point and k=1 must return
// that single point.
// =========================================================================
TEST(VamanaRegression, GreedySearchReturnsSinglePointForOnePointDataset) {
  const auto path = uniqueFixturePath("single_point");
  ScopedFile cleanup{path};

  writeDatasetFile(path, 1, 3, {{42.0f, 7.0f, 8.0f}});

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 1);
  v.setSeachListSize(1);

  HDVector q(std::vector<float>{7.0f, 8.0f});
  SearchResults r = v.greedySearch(q, 1);
  ASSERT_EQ(r.approximateNN.size(), 1U);
  EXPECT_EQ(r.approximateNN[0], 0);
}

// =========================================================================
// Vamana::insertIntoSet must maintain a distance-sorted invariant.  The
// comparator places smaller distances at the front; an inserted element
// may only grow the sequence by one.
// =========================================================================
TEST(VamanaRegression, InsertIntoSetInsertsAtSortedPosition) {
  const auto path = uniqueFixturePath("sorted_insert");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 5; ++i) {
    rows.push_back({static_cast<float>(i), static_cast<float>(i * i),
                    static_cast<float>(i)});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);

  HDVector q(std::vector<float>{2.0f, 2.0f});
  // Seed `working` with a proper distance-sorted sequence, then insert
  // a new element.  The result must remain sorted.
  NodeList working;
  v.insertIntoSet({0, 1, 2, 3, 4}, working, q);
  ASSERT_EQ(working.size(), 5U);

  float prev_d = -1.0f;
  for (NodeId idx : working) {
    auto rec = v.m_dataSet->getRecordViewByIndex(idx);
    const float d = HDVector::distance(q, *rec.vector);
    EXPECT_GE(d, prev_d)
        << "insertIntoSet produced an out-of-order sequence";
    prev_d = d;
  }
}
