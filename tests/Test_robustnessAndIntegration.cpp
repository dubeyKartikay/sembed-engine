// Robustness and integration regressions for deeper failure modes across
// validation, persistence, search behavior, and end-to-end recall.

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
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace {

std::filesystem::path uniqueFixturePath(const std::string &tag) {
  return testutils::uniqueFixturePath("robustness", tag);
}

using ScopedFile = testutils::ScopedPathCleanup;
using testutils::fixtureDir;
using testutils::writeDatasetFile;

std::vector<std::vector<float>> makeSmallClusteredRows(int64_t &outN,
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
// HDVector deeper bugs
// ============================================================================

// BUG: HDVector::distance uses float*float arithmetic before accumulating into
// the double running total. Large magnitude inputs therefore overflow to +Inf
// despite a double accumulator being available. A correct implementation would
// widen the per-dimension difference to double before squaring.
TEST(HDVectorRobustness, DistanceDoesNotOverflowOnLargeMagnitudeInputs) {
  HDVector big(std::vector<float>{1.0e20f, 0.0f});
  HDVector origin(std::vector<float>{0.0f, 0.0f});

  const float d = HDVector::distance(big, origin);
  EXPECT_TRUE(std::isfinite(d))
      << "distance returned " << d
      << " for well-representable inputs; float*float overflowed before "
         "widening to the double accumulator";
  EXPECT_NEAR(d, 1.0e20f, 1.0e15f)
      << "distance did not match the obvious ||(1e20, 0)|| = 1e20 answer";
}

// BUG: HDVector(int64_t) does not validate the dimension. A negative
// dimension is silently converted to an enormous unsigned size by
// std::vector, triggering bad_alloc at best and a multi-GB allocation attempt
// at worst. A defensive constructor should reject negative sizes up front.
TEST(HDVectorRobustness, NegativeDimensionConstructorIsRejected) {
  EXPECT_THROW((void)HDVector(-1), std::exception)
      << "constructing an HDVector with a negative dimension silently "
         "requests an enormous allocation instead of failing fast";
}

// BUG: when accumulating distances across many dimensions the implementation
// once again limits itself to float precision because the per-step multiply
// runs in float. With a large but finite number of sizeable dimensions the
// result drifts from the true value far more than float rounding alone would
// explain.
TEST(HDVectorRobustness, DistanceMatchesTheExpectedValueForMildlyLargeInputs) {
  const uint64_t dim = 16;
  std::vector<float> left(dim, 1.0e9f);
  std::vector<float> right(dim, 0.0f);
  HDVector a(left);
  HDVector b(right);

  // sqrt(sum_i (1e9)^2) = 1e9 * sqrt(16) = 4e9
  const float d = HDVector::distance(a, b);
  EXPECT_NEAR(d, 4.0e9f, 1.0e3f)
      << "distance should be 4e9 for 16 dims each contributing (1e9)^2";
}

// ============================================================================
// Graph deeper bugs
// ============================================================================

// BUG: Graph(0, R) calls getRandomNumber(0, -1) which evaluates gen() % 0 --
// that is division by zero and undefined behaviour / SIGFPE. An ANN graph
// over zero nodes should construct cleanly (even if it is useless), or at
// minimum throw a clean exception.
TEST(GraphRobustness, ZeroNodeGraphDoesNotCrashOrInvokeUB) {
  EXPECT_EXIT(
      {
        try {
          Graph g(0, 3);
          std::exit(0);
        } catch (const std::exception &) {
          std::exit(0);
        }
      },
      ::testing::ExitedWithCode(0), "");
}

// BUG: Graph(path) never checks whether the file opened successfully. If the
// file is missing the ifstream silently fails, numberOfNodes stays at zero,
// m_adj_list becomes empty, and getRandomNumber(0, -1) then divides by zero.
// A correct loader must report the I/O failure.
TEST(GraphRobustness, PathConstructorReportsMissingFileAsAnError) {
  const auto missing =
      fixtureDir() / "definitely_not_a_real_graph_file_9999.bin";
  std::error_code ec;
  std::filesystem::remove(missing, ec);

  EXPECT_EXIT(
      {
        try {
          Graph g(missing);
          // Loader silently accepted a missing file -- treat as failure.
          std::exit(1);
        } catch (const std::exception &) {
          std::exit(0);
        }
      },
      ::testing::ExitedWithCode(0), "");
}

// BUG: setOutNeighbors blindly stores whatever vector it is given, including
// a self edge. addOutNeighborUnique is careful to skip self-loops but the
// setter is not.  This lets buggy callers poison the adjacency structure.
TEST(GraphRobustness, SetOutNeighboursRejectsSelfLoop) {
  std::srand(0);
  Graph g(5, 2);
  EXPECT_THROW(g.setOutNeighbors(0, {0, 1}), std::invalid_argument)
      << "setOutNeighbors should reject self-loops instead of silently "
         "accepting them";
}

// BUG: Graph has no constructor that validates a negative degree threshold.
// Downstream, generateRandomNumbers(-1, n, i) executes `for (size_t i = 0;
// i < k; i++)` where k = -1 is promoted to size_t max, and the loop attempts
// roughly 2^64 iterations.  The loop is practically infinite.
// We put a short time-boxed smoke test behind an exit-test so the parent
// test binary survives.
TEST(GraphRobustness, ConstructorWithNegativeDegreeCompletesInReasonableTime) {
  EXPECT_EXIT(
      {
        // Self-destruct the child after 3 seconds so we never hang the
        // outer test runner if the implementation has an infinite loop on
        // negative degree. A SIGALRM kill will be reported as a non-zero
        // termination, which flags the bug.
        alarm(3);
        std::srand(0);
        try {
          Graph g(4, -1);
          std::exit(0);
        } catch (const std::exception &) {
          std::exit(0);
        }
      },
      ::testing::ExitedWithCode(0), "");
}

// BUG: the header validation only rejects storedDimensions < 1. A value of
// exactly 1 passes, which makes dimensions = 0 and yields a dataset of
// zero-dimensional vectors.  Such a dataset cannot meaningfully participate
// in an ANN index (every pair has distance zero).  The file format should
// reject this degenerate case up front.
TEST(DataSetRobustness, StoredDimensionsEqualToOneIsRejected) {
  const auto path = uniqueFixturePath("stored_dim_one");
  ScopedFile cleanup{path};

  int64_t n = 2;
  int64_t stored = 1;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(&stored), sizeof(stored));
  const float id0 = 100.0f;
  const float id1 = 200.0f;
  out.write(reinterpret_cast<const char *>(&id0), sizeof(id0));
  out.write(reinterpret_cast<const char *>(&id1), sizeof(id1));
  out.close();

  EXPECT_THROW(InMemoryDataSet ds(path), std::exception)
      << "storedDimensions==1 means the records have zero actual vector "
         "data; the loader should reject this degenerate file";
}

// BUG: InMemoryDataSet and FileDataSet should produce identical records when
// reading the same file.  The current code paths are independent and prone
// to silently diverging -- this test pins down that they agree.
TEST(DataSetRobustness, InMemoryAndFileDataSetAgreeOnTheSameFile) {
  const auto path = uniqueFixturePath("in_mem_vs_file");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows = {
      {10.0f, 1.0f, 2.0f, 3.0f},
      {20.0f, 4.0f, 5.0f, 6.0f},
      {30.0f, 7.0f, 8.0f, 9.0f},
  };
  writeDatasetFile(path, rows.size(), 4, rows);

  InMemoryDataSet inMem(path);
  FileDataSet onDisk(path);

  ASSERT_EQ(inMem.getN(), onDisk.getN());
  ASSERT_EQ(inMem.getDimensions(), onDisk.getDimensions());

  for (uint64_t i = 0; i < inMem.getN(); ++i) {
    auto a = inMem.getRecordViewByIndex(i);
    auto b = onDisk.getRecordViewByIndex(i);
    ASSERT_NE(a.vector, nullptr);
    ASSERT_NE(b.vector, nullptr);
    EXPECT_EQ(a.recordId, b.recordId);
    ASSERT_EQ(a.vector->getDimension(), b.vector->getDimension());
    for (uint64_t d = 0; d < a.vector->getDimension(); ++d) {
      EXPECT_FLOAT_EQ((*a.vector)[d], (*b.vector)[d])
          << "in-memory and on-disk loaders disagree at row " << i
          << " dim " << d;
    }
  }
}

// BUG: FileDataSet does not close its file on failed construction.  If the
// header check rejects the file (storedDimensions < 1), m_file remains open
// and leaks the underlying handle.  Constructing many bad datasets in a
// tight loop would exhaust the process file descriptor table.  We cannot
// observe the handle directly from user code, so instead we assert that
// repeated construction failures are themselves reproducible, which they
// should be for a correctly-cleaning-up implementation.
TEST(DataSetRobustness, RepeatedFailedConstructionStaysReproducible) {
  const auto path = uniqueFixturePath("bad_header");
  ScopedFile cleanup{path};
  int64_t n = 1;
  int64_t stored = 0; // invalid -- loader rejects
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(&stored), sizeof(stored));
  out.close();

  for (uint64_t attempt = 0; attempt < 8; ++attempt) {
    EXPECT_THROW(FileDataSet ds(path), std::exception)
        << "attempt " << attempt
        << ": loader stopped rejecting invalid files after earlier failures, "
           "suggesting a stale file handle or cached state";
  }
}

// BUG: RecordView vectors produced by getNRecordViewsFromIndex and
// getNHDVectorsFromIndex must share the same data with
// getRecordViewByIndex.  Any divergence is a serialisation or copy bug.
TEST(DataSetRobustness, RangedAndSingleLookupsMatch) {
  const auto path = uniqueFixturePath("ranged_vs_single");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows = {
      {7.0f, 1.0f, -1.0f},
      {8.0f, 2.0f, -2.0f},
      {9.0f, 3.0f, -3.0f},
      {10.0f, 4.0f, -4.0f},
  };
  writeDatasetFile(path, rows.size(), 3, rows);

  InMemoryDataSet ds(path);
  auto range = ds.getNRecordViewsFromIndex(1, 2);
  auto vectorsOnly = ds.getNHDVectorsFromIndex(1, 2);
  ASSERT_NE(range, nullptr);
  ASSERT_NE(vectorsOnly, nullptr);
  ASSERT_EQ(range->size(), 2U);
  ASSERT_EQ(vectorsOnly->size(), 2U);

  for (uint64_t offset = 0; offset < 2; ++offset) {
    const int64_t absolute = static_cast<int64_t>(1 + offset);
    auto single = ds.getRecordViewByIndex(absolute);
    ASSERT_NE(single.vector, nullptr);
    ASSERT_NE(range->at(offset).vector, nullptr);
    ASSERT_NE(vectorsOnly->at(offset), nullptr);

    EXPECT_EQ(single.recordId, range->at(offset).recordId);
    for (uint64_t d = 0; d < single.vector->getDimension(); ++d) {
      EXPECT_FLOAT_EQ((*single.vector)[d], (*range->at(offset).vector)[d]);
      EXPECT_FLOAT_EQ((*single.vector)[d], (*vectorsOnly->at(offset))[d]);
    }
  }
}

// ============================================================================
// Vamana deeper bugs
// ============================================================================

// BUG: Constructing a Vamana on a dataset with zero records is catastrophic
// because Graph(0, R) then evaluates gen() % 0 in getRandomNumber and
// crashes.  Either the Vamana or the Graph should detect the degenerate
// case up front.
TEST(VamanaRobustness, ConstructorWithEmptyDatasetDoesNotCrash) {
  const auto path = uniqueFixturePath("empty_dataset");
  ScopedFile cleanup{path};
  int64_t n = 0;
  int64_t stored = 3;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(&stored), sizeof(stored));
  out.close();

  EXPECT_EXIT(
      {
        try {
          auto ds = std::make_unique<InMemoryDataSet>(path);
          Vamana v(std::move(ds), 2);
          std::exit(0);
        } catch (const std::exception &) {
          std::exit(0);
        }
      },
      ::testing::ExitedWithCode(0), "");
}

// BUG: buildIndex() prints "Making <node>" for every node it processes.  A
// library routine must not spam stdout -- consumers get their output
// polluted and test logs become unreadable.
TEST(VamanaRobustness, BuildIndexDoesNotSpamStdout) {
  const auto path = uniqueFixturePath("silent_build");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t stored = 0;
  const auto rows = makeSmallClusteredRows(n, stored);
  writeDatasetFile(path, n, stored, rows);

  std::stringstream captured;
  std::streambuf *prev = std::cout.rdbuf(captured.rdbuf());

  {
    std::srand(0);
    auto ds = std::make_unique<InMemoryDataSet>(path);
    Vamana v(std::move(ds), 3);
    (void)v;
  }

  std::cout.rdbuf(prev);
  EXPECT_TRUE(captured.str().empty())
      << "buildIndex() wrote the following to stdout:\n" << captured.str();
}


// BUG: a query whose dimension does not match the dataset must be rejected
// before the search starts.  Currently the algorithm happily pushes the
// medoid into the result before running distance, and the error message
// surfaces only when the algorithm tries to compute a distance -- at that
// point callers have already paid part of the search cost.  Even worse, if
// the medoid has no out-neighbours the search silently returns the medoid
// without ever checking the dimension.
TEST(VamanaRobustness, GreedySearchRejectsMismatchedQueryDimensionImmediately) {
  const auto path = uniqueFixturePath("dim_mismatch");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t stored = 0;
  const auto rows = makeSmallClusteredRows(n, stored);
  writeDatasetFile(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);
  v.setSearchListSize(4);

  // Dataset vectors are 2-dimensional; this query has 5 dimensions.
  HDVector bad(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  EXPECT_THROW((void)v.greedySearch(bad, 1), std::invalid_argument)
      << "greedySearch accepted a query whose dimension did not match the "
         "dataset";
}

// BUG: greedySearch must never return indices that are not valid dataset
// records.  A subtle off-by-one or uninitialised adjacency entry would leak
// garbage indices to the caller.
TEST(VamanaRobustness, GreedySearchResultsAreWithinDatasetBounds) {
  const auto path = uniqueFixturePath("in_bounds");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t stored = 0;
  const auto rows = makeSmallClusteredRows(n, stored);
  writeDatasetFile(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  const uint64_t datasetSize = static_cast<uint64_t>(n);
  Vamana v(std::move(ds), 3);
  v.setSearchListSize(static_cast<int64_t>(datasetSize));

  HDVector q(std::vector<float>{12.5f, 7.5f});
  SearchResults r = v.greedySearch(q, 5);

  for (NodeId idx : r.approximateNN) {
    EXPECT_GE(idx, 0);
    EXPECT_LT(static_cast<uint64_t>(idx), datasetSize);
  }
  for (NodeId idx : r.visited) {
    EXPECT_GE(idx, 0);
    EXPECT_LT(static_cast<uint64_t>(idx), datasetSize);
  }
}

// BUG: setDistanceThreshold stores alpha but it must then show up in
// isToBePruned. Flipping the threshold between calls should visibly change
// the pruning decision. If it does not, the setter is broken.
TEST(VamanaRobustness, SetDistanceThresholdChangesPruningDecision) {
  const auto path = uniqueFixturePath("alpha_setter");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows = {
      {0.0f, 0.0f, 0.0f},
      {1.0f, 1.0f, 0.0f},
      {2.0f, 2.0f, 0.0f},
      {3.0f, 3.0f, 0.0f},
  };
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);

  // Pick three distinct nodes so that distance(p, p_dash) and
  // distance(p_star, p_dash) are both positive but different.
  // alpha = 0.5:  0.5 * d(p_star, p_dash) <= d(p, p_dash) ?
  v.setDistanceThreshold(0.5f);
  const bool atHalf = v.isToBePruned(/*p_dash=*/3, /*p_star=*/1, /*p=*/0);

  // alpha = 10.0: 10 * d(p_star, p_dash) <= d(p, p_dash) ?
  v.setDistanceThreshold(10.0f);
  const bool atTen = v.isToBePruned(/*p_dash=*/3, /*p_star=*/1, /*p=*/0);

  EXPECT_NE(atHalf, atTen)
      << "isToBePruned returned the same answer under alpha=0.5 and "
         "alpha=10, so setDistanceThreshold has no effect";
}

// BUG: construction currently depends on shared process-global randomness, so
// building two indexes back to back can quietly entangle them. Callers expect
// deterministic, instance-local construction for identical inputs.
TEST(VamanaRobustness, InstancesDoNotLeakRandomStateBetweenConstructions) {
  const auto path = uniqueFixturePath("leaky_rand");
  ScopedFile cleanup{path};

  int64_t n = 0;
  int64_t stored = 0;
  const auto rows = makeSmallClusteredRows(n, stored);
  writeDatasetFile(path, n, stored, rows);

  auto neighboursOf = [&path](NodeId node) {
    std::srand(0);
    auto ds = std::make_unique<InMemoryDataSet>(path);
    Vamana v(std::move(ds), 3);
    NodeList neighbours = v.getOutNeighbors(node);
    std::sort(neighbours.begin(), neighbours.end());
    return neighbours;
  };

  const auto withReset = neighboursOf(0);

  // Perform a second construction without resetting std::rand first. In a
  // library that owns its own construction-time randomness, this must still
  // yield identical results. A process-global dependency would make the two
  // calls diverge.
  auto neighboursOfNoReset = [&path](NodeId node) {
    auto ds = std::make_unique<InMemoryDataSet>(path);
    Vamana v(std::move(ds), 3);
    NodeList neighbours = v.getOutNeighbors(node);
    std::sort(neighbours.begin(), neighbours.end());
    return neighbours;
  };

  // First, warm up global rand so the no-reset call is visibly downstream of
  // an unrelated consumer.
  for (uint64_t i = 0; i < 25; ++i) {
    (void)std::rand();
  }
  const auto withoutReset = neighboursOfNoReset(0);

  EXPECT_EQ(withReset, withoutReset)
      << "adjacency for node 0 depends on shared process randomness rather "
         "than deterministic instance-local construction";
}

// BUG: insertIntoSet must insert every element from `from` that isn't already
// present in `to`. Failing to do so silently shrinks the visited set used by
// Vamana's greedySearch and prune routines, degrading recall.
TEST(VamanaRobustness, InsertIntoSetInsertsEveryMissingElement) {
  const auto path = uniqueFixturePath("insert_every");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows = {
      {0.0f, 0.0f, 0.0f},
      {1.0f, 1.0f, 1.0f},
      {2.0f, 2.0f, 2.0f},
      {3.0f, 3.0f, 3.0f},
      {4.0f, 4.0f, 4.0f},
      {5.0f, 5.0f, 5.0f},
  };
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);

  HDVector q(std::vector<float>{0.5f, 0.5f});
  NodeList to;
  v.insertIntoSet({2, 4}, to, q);
  v.insertIntoSet({1, 3, 5}, to, q);

  std::unordered_set<NodeId> unique(to.begin(), to.end());
  const std::unordered_set<NodeId> expected = {1, 2, 3, 4, 5};
  EXPECT_EQ(unique, expected)
      << "insertIntoSet dropped at least one element; got "
      << ::testing::PrintToString(to);
}

// BUG: insertIntoSet must keep `to` sorted by distance from the query. The
// greedySearch algorithm relies on lower_bound finding the right insertion
// point, and that only works if the invariant is honoured.
TEST(VamanaRobustness, InsertIntoSetKeepsToSortedByDistance) {
  const auto path = uniqueFixturePath("insert_sorted");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows = {
      {0.0f, 0.0f, 0.0f},  {1.0f, 1.0f, 0.0f},  {2.0f, 4.0f, 0.0f},
      {3.0f, 9.0f, 0.0f},  {4.0f, 16.0f, 0.0f}, {5.0f, 25.0f, 0.0f},
  };
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);

  HDVector q(std::vector<float>{0.0f, 0.0f});

  NodeList to;
  v.insertIntoSet({5, 3, 1, 4, 2, 0}, to, q);

  for (size_t i = 1; i < to.size(); ++i) {
    const float before = HDVector::distance(
        q, *v.getRecordViewByIndex(to[i - 1]).vector);
    const float after = HDVector::distance(
        q, *v.getRecordViewByIndex(to[i]).vector);
    EXPECT_LE(before, after)
        << "insertIntoSet produced an out-of-order pair at positions "
        << (i - 1) << " and " << i;
  }
}

// ============================================================================
// Utils deeper bugs
// ============================================================================

// BUG: generateRandomNumbers(k, 0, blackList) executes rand() % 0, which is
// undefined behaviour. The function must either guard against a zero range
// or return immediately with an empty vector.
TEST(UtilsRobustness, GenerateRandomNumbersWithZeroNDoesNotInvokeUB) {
  EXPECT_EXIT(
      {
        try {
          std::srand(7);
          const auto result = generateRandomNumbers(3, 0, /*blackList=*/-1);
          // A correct implementation should produce an empty vector.
          if (!result.empty()) {
            std::exit(1);
          }
          std::exit(0);
        } catch (const std::exception &) {
          std::exit(0);
        }
      },
      ::testing::ExitedWithCode(0), "");
}

// BUG: When k > n the function cannot possibly produce k unique values, yet
// the contract is ambiguous today. A stable API should return min(k, n) -- or
// at worst throw -- but never exceed n unique outputs.
TEST(UtilsRobustness, GenerateRandomNumbersCannotReturnMoreThanNUniqueValues) {
  std::srand(42);
  const uint64_t n = 4;
  const uint64_t k = 10;

  const auto result = generateRandomNumbers(k, n, /*blackList=*/-1);
  std::unordered_set<NodeId> unique(result.begin(), result.end());

  EXPECT_LE(unique.size(), static_cast<size_t>(n));
  EXPECT_LE(result.size(), static_cast<size_t>(n))
      << "generateRandomNumbers returned " << result.size()
      << " entries from a population of only " << n;
}

// BUG: getPermutation must always produce a permutation of {0..n-1}. Even at
// n == 0 the behaviour must be defined.  Today the implementation happens to
// work for n == 0 but there is no test pinning the contract down.
TEST(UtilsRobustness, GetPermutationZeroProducesEmptyResult) {
  const auto perm = getPermutation(0);
  EXPECT_TRUE(perm.empty());
}

// BUG: isValidPath's contract is ambiguous.  It is used to gate the
// FileDataSet constructor but it reports directories as "valid", which
// then falls into a generic runtime_error from fstream::open.  A loader
// helper should reject non-regular-file paths up front.
TEST(UtilsRobustness, IsValidPathRejectsDirectoryInputs) {
  const auto dir = fixtureDir();
  ASSERT_TRUE(std::filesystem::is_directory(dir));
  EXPECT_FALSE(isValidFile(dir.string()))
      << "isValidPath treats a directory as a valid dataset path";
}

// BUG: Empty strings must not be mistaken for existing files. This protects
// against accidentally reading whatever happens to exist at the current
// working directory.
TEST(UtilsRobustness, IsValidPathRejectsEmptyString) {
  EXPECT_FALSE(isValidPath(""));
}


// BUG: generateRandomNumbers must use a *shared* or caller-provided source of
// randomness so successive calls produce different sequences.  The fact that
// std::srand is consulted means the caller is responsible for seeding it, but
// there is still an expectation that, once seeded, two back-to-back calls do
// not return identical vectors.  A broken implementation (for example one
// that reseeded internally) would always return the same set.
TEST(UtilsRobustness, GenerateRandomNumbersDoesNotReturnIdenticalBackToBack) {
  std::srand(2025);
  const auto first = generateRandomNumbers(4, 32, /*blackList=*/-1);
  const auto second = generateRandomNumbers(4, 32, /*blackList=*/-1);
  ASSERT_EQ(first.size(), 4U);
  ASSERT_EQ(second.size(), 4U);
  EXPECT_NE(first, second)
      << "two back-to-back generateRandomNumbers calls returned identical "
         "vectors -- the generator is effectively stuck";
}

// ============================================================================
// End-to-end / integration bugs
// ============================================================================

// BUG: on a handful of tightly-clustered points with L == N, greedySearch
// should recover the brute-force top-k exactly. Any deviation reveals a
// defect in the traversal, pruning, or sorting logic.
TEST(IntegrationRegression, RecallIsPerfectAtLEqualsNOnMediumDataset) {
  const auto path = uniqueFixturePath("full_L_recall");
  ScopedFile cleanup{path};

  // 30 points in three well-separated clusters.
  std::vector<std::vector<float>> rows;
  int64_t id = 0;
  for (uint64_t cluster = 0; cluster < 3; ++cluster) {
    for (uint64_t i = 0; i < 10; ++i) {
      const float base = static_cast<float>(cluster * 100);
      rows.push_back({static_cast<float>(id), base + static_cast<float>(i),
                      base - static_cast<float>(i) * 0.25f});
      ++id;
    }
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 6);
  v.setSearchListSize(static_cast<int64_t>(rows.size()));

  const std::vector<float> query = {202.5f, -0.625f};
  HDVector q(query);

  auto squared = [](const std::vector<float> &payload,
                    const std::vector<float> &qv) {
    float total = 0.0f;
    for (size_t d = 0; d < payload.size(); ++d) {
      const float delta = payload[d] - qv[d];
      total += delta * delta;
    }
    return total;
  };

  std::vector<std::pair<float, int64_t>> ranked;
  ranked.reserve(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    const std::vector<float> payload(rows[i].begin() + 1, rows[i].end());
    ranked.push_back({squared(payload, query), static_cast<int64_t>(i)});
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const auto &l, const auto &r) {
              if (l.first == r.first) {
                return l.second < r.second;
              }
              return l.first < r.first;
            });

  SearchResults approx = v.greedySearch(q, 5);
  ASSERT_EQ(approx.approximateNN.size(), 5U);

  NodeList expected;
  expected.reserve(5);
  for (uint64_t i = 0; i < 5; ++i) {
    expected.push_back(static_cast<NodeId>(ranked[i].second));
  }
  EXPECT_EQ(approx.approximateNN, expected)
      << "with L == N the top-5 ANN should equal the brute-force top-5";
}

// BUG: running the entire dataset as queries against itself must recover
// every point exactly. This is the simplest non-trivial recall guarantee any
// ANN library should honour.
TEST(IntegrationRegression, SelfRecallIsPerfectAtLEqualsN) {
  const auto path = uniqueFixturePath("self_recall");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (uint64_t i = 0; i < 25; ++i) {
    const float x = static_cast<float>(i);
    rows.push_back({static_cast<float>(i), x, x * 0.5f});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 5);
  v.setSearchListSize(static_cast<int64_t>(rows.size()));

  uint64_t misses = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    const std::vector<float> payload(rows[i].begin() + 1, rows[i].end());
    HDVector q(payload);
    SearchResults r = v.greedySearch(q, 1);
    if (r.approximateNN.size() != 1U ||
        r.approximateNN.front() != static_cast<int64_t>(i)) {
      ++misses;
    }
  }
  EXPECT_EQ(misses, 0U)
      << "self-query failed to return the original point for " << misses
      << " records";
}
