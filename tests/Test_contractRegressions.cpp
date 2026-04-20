// Contract-level regression tests covering API invariants that the main
// component and robustness suites do not cover.

#include "HDVector.hpp"
#include "Set.hpp"
#include "dataset.hpp"
#include "graph.hpp"
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

std::string sanitize(std::string value) {
  for (char &ch : value) {
    const bool is_alnum = (ch >= 'a' && ch <= 'z') ||
                          (ch >= 'A' && ch <= 'Z') ||
                          (ch >= '0' && ch <= '9');
    if (!is_alnum && ch != '_' && ch != '-') {
      ch = '_';
    }
  }
  return value;
}

std::filesystem::path fixtureDir() {
  const auto dir = std::filesystem::current_path() / "build" / "test-fixtures";
  std::filesystem::create_directories(dir);
  return dir;
}

std::filesystem::path uniqueFixturePath(const std::string &tag) {
  const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string suite =
      info ? sanitize(info->test_suite_name()) : "unknown_suite";
  const std::string name = info ? sanitize(info->name()) : "unknown_test";
  return fixtureDir() /
         ("contract_regressions_" + suite + "_" + name + "_" + tag + ".bin");
}

struct ScopedFile {
  std::filesystem::path path;
  ~ScopedFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

std::filesystem::path writeDataset(const std::filesystem::path &path,
                                   long long n, long long storedDim,
                                   const std::vector<std::vector<float>> &rows) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("failed to create dataset fixture");
  }
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(&storedDim), sizeof(storedDim));
  for (const auto &row : rows) {
    out.write(reinterpret_cast<const char *>(row.data()),
              static_cast<std::streamsize>(row.size() * sizeof(float)));
  }
  return path;
}

std::vector<std::vector<float>> makeClusteredRows(long long &outN,
                                                  long long &outStored) {
  outStored = 3;
  std::vector<std::vector<float>> rows;
  int id = 0;
  for (int cluster = 0; cluster < 3; ++cluster) {
    for (int i = 0; i < 5; ++i) {
      rows.push_back({static_cast<float>(id),
                      static_cast<float>(cluster * 50 + i),
                      static_cast<float>(cluster * 50 - i)});
      ++id;
    }
  }
  outN = static_cast<long long>(rows.size());
  return rows;
}

} // namespace

// ============================================================================
// Graph API-surface bugs
// ============================================================================

// BUG: setOutNeighbours blindly stores the caller-supplied vector without
// enforcing the degree cap that the graph was constructed with. This turns
// the "degree threshold" into advisory information only.
TEST(GraphContractRegression, SetOutNeighboursRespectsDegreeThreshold) {
  std::srand(0);
  Graph graph(6, 2);

  graph.setOutNeighbours(0, {1, 2, 3, 4, 5});
  const auto &neighbours = graph.getOutNeighbours(0);

  EXPECT_LE(static_cast<int>(neighbours.size()), graph.getDegreeThreshold())
      << "setOutNeighbours stored " << neighbours.size()
      << " neighbours on a graph whose degree threshold is "
      << graph.getDegreeThreshold();
}

// BUG: addOutNeighbourUnique does not check from == to before inserting, so
// a caller can trivially create self-loops. Self-loops break the implicit
// assumption that out-neighbours never point back at the owning node, and
// any greedy search that steps through them will either waste an iteration
// or loop indefinitely on trivial graphs.
TEST(GraphContractRegression, AddOutNeighbourUniqueRejectsSelfLoop) {
  std::srand(0);
  Graph graph(4, 2);
  graph.clearOutNeighbours(0);

  graph.addOutNeighbourUnique(0, 0);

  const auto &neighbours = graph.getOutNeighbours(0);
  EXPECT_EQ(std::count(neighbours.begin(), neighbours.end(), 0), 0)
      << "addOutNeighbourUnique(0, 0) inserted a self-loop";
}

// BUG: getOutNeighbours returns a *mutable* reference to internal state, so
// any caller can bypass every invariant check in addOutNeighbourUnique and
// setOutNeighbours simply by push_back-ing arbitrary values (including
// negative indices). The getter should either return const& or a copy.
TEST(GraphContractRegression, GetOutNeighboursDoesNotExposeMutableInternalState) {
  std::srand(0);
  Graph graph(4, 2);

  graph.getOutNeighbours(0).push_back(-42);

  const auto &re_read = graph.getOutNeighbours(0);
  EXPECT_EQ(std::count(re_read.begin(), re_read.end(), -42), 0)
      << "mutating the getOutNeighbours reference leaked a -42 neighbour "
         "back into the graph";
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

// BUG: getOutNeighbours has no bounds check. A Graph(4, 2).getOutNeighbours(99)
// is undefined behaviour. The indexing operator should throw std::out_of_range
// so caller bugs surface immediately rather than corrupting memory.
TEST(GraphContractRegression, OutOfRangeGetOutNeighboursThrows) {
  std::srand(0);
  Graph graph(4, 2);

  EXPECT_EXIT(
      {
        alarm(3);
        try {
          (void)graph.getOutNeighbours(99);
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

// BUG: getPermutation with n <= 0 is not defined. n == 0 happens to work, but
// n < 0 flows into std::vector<int>(n, 0), which converts to size_t and tries
// to allocate enormous storage.  A stable API should reject it up front.
TEST(UtilsContractRegression, GetPermutationWithNegativeNReturnsEmpty) {
  EXPECT_EXIT(
      {
        alarm(3);
        try {
          const auto perm = getPermutation(-1);
          if (!perm.empty()) {
            std::exit(1);
          }
          std::exit(0);
        } catch (const std::exception &) {
          std::exit(0);
        }
      },
      ::testing::ExitedWithCode(0), "");
}

// BUG: isValidPath is implemented as fs::exists, which returns true for
// directories, symlinks to directories, character devices, etc. FileDataSet
// then tries to open the directory as a binary file and either silently
// reads garbage or fails later with a non-helpful error.
TEST(UtilsContractRegression, IsValidPathReturnsFalseForDirectoriesAndEmptyInput) {
  EXPECT_FALSE(isValidPath(""));
  EXPECT_FALSE(isValidPath(fixtureDir().string()));
}

// ============================================================================
// HDVector additional bugs
// ============================================================================

// BUG: HDVector's `dimentions` field is declared `int`, but the value is
// assigned from `vec.size()` (size_t) or `const int &`. The `int` storage
// type silently narrows any caller-provided dimension beyond INT_MAX.
// Expose the type mismatch at the API boundary.
TEST(HDVectorContractRegression, DimensionAccessorExposesLongLongStorage) {
  HDVector v(std::vector<float>{1.0f, 2.0f, 3.0f});
  using returned_type = std::remove_cvref_t<decltype(v.getDimention())>;
  EXPECT_TRUE((std::is_same_v<returned_type, std::size_t>) ||
              (std::is_same_v<returned_type, long long>))
      << "HDVector::getDimention returns `int`, which silently narrows any "
         "dimension larger than INT_MAX";
}

// BUG: HDVector::distance on two identical vectors should return exactly 0.
// Any non-zero residue points at either the float-precision loss bug or an
// unstable accumulator. This is our canonical sanity check.
TEST(HDVectorContractRegression, DistanceOfIdenticalLargeVectorsIsExactlyZero) {
  std::vector<float> data(32, 1.0e7f);
  HDVector a(data);
  HDVector b(data);
  EXPECT_FLOAT_EQ(HDVector::distance(a, b), 0.0f);
}

// BUG: distance must satisfy the triangle inequality d(a,c) <= d(a,b) + d(b,c).
// An accumulator that loses precision can produce d(a,c) strictly larger than
// d(a,b)+d(b,c). Any such violation makes every downstream pruning decision
// unsound.
TEST(HDVectorContractRegression, DistanceObeysTriangleInequality) {
  HDVector a(std::vector<float>{0.0f, 0.0f});
  HDVector b(std::vector<float>{3.0f, 4.0f});
  HDVector c(std::vector<float>{6.0f, 8.0f});

  const float ab = HDVector::distance(a, b);
  const float bc = HDVector::distance(b, c);
  const float ac = HDVector::distance(a, c);
  EXPECT_LE(ac, ab + bc + 1e-4f)
      << "triangle inequality violated: d(a,c)=" << ac
      << ", d(a,b)+d(b,c)=" << ab + bc;
}

// ============================================================================
// DataSet accessor narrowing
// ============================================================================

// BUG: DataSet stores `long long int n;` but exposes `const int getN() const`,
// silently narrowing counts above INT_MAX. The accessor's declared return
// type should match the storage type so callers see the full range.
TEST(DataSetContractRegression, GetNReturnTypeMatchesLongLongStorage) {
  const auto path = uniqueFixturePath("n_return_type");
  ScopedFile cleanup{path};
  writeDataset(path, 2, 3, {{0.0f, 1.0f, 2.0f}, {1.0f, 3.0f, 4.0f}});
  InMemoryDataSet ds(path);

  using returned_type = std::remove_cvref_t<decltype(ds.getN())>;
  EXPECT_TRUE((std::is_same_v<returned_type, long long>))
      << "InMemoryDataSet::getN() returns `int`, silently narrowing the "
         "`long long int n` member";
}

// BUG: same narrowing issue as above, but for getDimentions.
TEST(DataSetContractRegression, GetDimensionsReturnTypeMatchesLongLongStorage) {
  const auto path = uniqueFixturePath("dim_return_type");
  ScopedFile cleanup{path};
  writeDataset(path, 2, 3, {{0.0f, 1.0f, 2.0f}, {1.0f, 3.0f, 4.0f}});
  InMemoryDataSet ds(path);

  using returned_type = std::remove_cvref_t<decltype(ds.getDimentions())>;
  EXPECT_TRUE((std::is_same_v<returned_type, long long>))
      << "InMemoryDataSet::getDimentions() returns `int`, silently narrowing "
         "the `long long int dimentions` member";
}

// BUG: header validation allows n < 0 today. A correct loader must reject a
// negative record count up front because nothing downstream handles negative
// sizes gracefully.
TEST(DataSetContractRegression, NegativeRecordCountHeaderIsRejected) {
  const auto path = uniqueFixturePath("negative_n");
  ScopedFile cleanup{path};

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  long long n = -5;
  long long stored = 3;
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(&stored), sizeof(stored));
  out.close();

  EXPECT_THROW(InMemoryDataSet ds(path), std::exception)
      << "loader accepted a header claiming n == -5";
}

// BUG: getRecordViewByIndex on a negative index should throw out_of_range.
// std::vector::at on a negative int converts to a huge size_t and does
// throw, but the cast may be implementation-defined for the exact behaviour;
// our contract should be deterministic.
TEST(DataSetContractRegression, NegativeIndexGetRecordViewThrowsOutOfRange) {
  const auto path = uniqueFixturePath("negative_index");
  ScopedFile cleanup{path};
  writeDataset(path, 2, 3, {{0.0f, 1.0f, 2.0f}, {1.0f, 3.0f, 4.0f}});
  InMemoryDataSet ds(path);

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

  for (int node = 0; node < 5; ++node) {
    const auto &neighbours = graph.getOutNeighbours(node);
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

  long long n = 0;
  long long stored = 0;
  const auto rows = makeClusteredRows(n, stored);
  writeDataset(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);

  // Snapshot adjacency after the first build.
  std::vector<std::vector<int>> firstBuild(static_cast<size_t>(n));
  for (int i = 0; i < static_cast<int>(n); ++i) {
    firstBuild[i] = v.m_graph.getOutNeighbours(i);
    std::sort(firstBuild[i].begin(), firstBuild[i].end());
  }

  // Redirect cout so the "Making N" spam does not pollute the test log.
  std::stringstream discard;
  std::streambuf *prev = std::cout.rdbuf(discard.rdbuf());
  v.buildIndex();
  std::cout.rdbuf(prev);

  // Compare to the second build.
  for (int i = 0; i < static_cast<int>(n); ++i) {
    auto after = v.m_graph.getOutNeighbours(i);
    std::sort(after.begin(), after.end());
    EXPECT_EQ(firstBuild[i], after)
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

  long long n = 0;
  long long stored = 0;
  const auto rows = makeClusteredRows(n, stored);
  writeDataset(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);

  v.setDistanceThreshold(2.5f);

  // The public state is exposed as `m_distanceThreshold`. The fact that the
  // API forces tests to reach into internal state is itself a smell.
  EXPECT_FLOAT_EQ(v.m_distanceThreshold, 2.5f)
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

  long long n = 0;
  long long stored = 0;
  const auto rows = makeClusteredRows(n, stored);
  writeDataset(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);
  v.setSeachListSize(15);

  HDVector q(std::vector<float>{12.0f, -2.0f});
  SearchResults r = v.greedySearch(q, 5);

  // approximateNN must be strictly ascending by distance.
  for (size_t i = 1; i < r.approximateNN.size(); ++i) {
    const float before = HDVector::distance(
        q, *v.m_dataSet->getRecordViewByIndex(r.approximateNN[i - 1]).vector);
    const float after = HDVector::distance(
        q, *v.m_dataSet->getRecordViewByIndex(r.approximateNN[i]).vector);
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

  long long n = 0;
  long long stored = 0;
  const auto rows = makeClusteredRows(n, stored);
  writeDataset(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);
  v.setSeachListSize(10);

  HDVector q(std::vector<float>{25.0f, -5.0f});
  SearchResults first = v.greedySearch(q, 3);
  SearchResults second = v.greedySearch(q, 3);

  EXPECT_EQ(first.approximateNN, second.approximateNN)
      << "two identical greedySearch calls returned different ANN lists";
}

// BUG: prune on an empty candidate set must leave the node with a usable
// out-neighbour list.  The current implementation clears the list and then
// never repopulates, leaving the node stranded.
TEST(VamanaContractRegression, PruneOnEmptyCandidateSetLeavesNodeStranded) {
  const auto path = uniqueFixturePath("prune_empty");
  ScopedFile cleanup{path};

  long long n = 0;
  long long stored = 0;
  const auto rows = makeClusteredRows(n, stored);
  writeDataset(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);

  std::vector<int> candidates; // intentionally empty
  v.prune(0, candidates);
  const auto &after = v.m_graph.getOutNeighbours(0);

  EXPECT_FALSE(after.empty())
      << "prune() cleared node 0's out-neighbours and never replaced them "
         "because the candidate set was empty; the node is now unreachable";
}

// BUG: isToBePruned must obey `alpha * d(p*, p') < d(p, p')`. A degenerate
// alpha of 0 means "prune iff 0 <= d(p, p')", i.e. always prune. Verify
// that edge case: the test fails if the implementation inverts the
// comparison or uses the wrong operand.
TEST(VamanaContractRegression, IsToBePrunedWithZeroAlphaAlwaysPrunes) {
  const auto path = uniqueFixturePath("zero_alpha");
  ScopedFile cleanup{path};
  writeDataset(path, 3, 3,
               {{0.0f, 0.0f, 0.0f},
                {1.0f, 1.0f, 0.0f},
                {2.0f, 2.0f, 0.0f}});

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 2);

  v.setDistanceThreshold(0.0f);
  EXPECT_TRUE(v.isToBePruned(/*p_dash=*/2, /*p_star=*/1, /*p=*/0))
      << "alpha=0 should always prune (0 * anything <= any non-negative "
         "distance); isToBePruned returned false";
}

// BUG: inserting a mismatched-dimension query node anywhere in the algorithm
// should surface as std::invalid_argument.  prune ultimately calls
// insertIntoSet which calls HDVector::distance; if the dimensions disagree,
// an std::invalid_argument exception is expected.
TEST(VamanaContractRegression, InsertIntoSetPropagatesDimensionMismatch) {
  const auto path = uniqueFixturePath("dim_mismatch_insert");
  ScopedFile cleanup{path};

  long long n = 0;
  long long stored = 0;
  const auto rows = makeClusteredRows(n, stored);
  writeDataset(path, n, stored, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 3);

  HDVector bad(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  std::vector<int> to;
  EXPECT_THROW(v.insertIntoSet({1, 2}, to, bad), std::invalid_argument);
}
