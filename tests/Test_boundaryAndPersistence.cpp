// Boundary-condition and persistence regression tests for dataset and graph
// loading, mutation, and serialization behavior.

#include "HDVector.hpp"
#include "dataset.hpp"
#include "graph.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace {

struct DatasetFixtureData {
  int64_t n = 4;
  int64_t dimensions = 3;
  std::vector<std::vector<float>> rows = {
      {10.0f, 1.0f, 2.0f},
      {20.0f, 3.0f, 4.0f},
      {30.0f, 5.0f, 6.0f},
      {40.0f, 7.0f, 8.0f},
  };
};

template <typename DataSetType>
class DataSetBoundaryTest : public ::testing::Test {
protected:
  DatasetFixtureData fixture;
  std::filesystem::path datasetPath;

  void SetUp() override {
    datasetPath = testutils::uniqueFixturePath("boundary", "dataset");
    testutils::writeDatasetFile(datasetPath, fixture.n, fixture.dimensions,
                                fixture.rows);
  }

  void TearDown() override {
    testutils::ScopedPathCleanup cleanup(datasetPath);
  }

  std::unique_ptr<DataSetType> makeDataSet() {
    return std::make_unique<DataSetType>(datasetPath);
  }
};

using DataSetImplementations = ::testing::Types<FlatDataSet>;
TYPED_TEST_SUITE(DataSetBoundaryTest, DataSetImplementations);

TYPED_TEST(DataSetBoundaryTest, AllowsEmptyRangesAtDatasetBoundary) {
  auto dataSet = this->makeDataSet();

  auto records = dataSet->getRecordViewsFromIndex(this->fixture.rows.size(), 0);
  // vector return value is always materialized.
  EXPECT_TRUE(records.empty());
}

TYPED_TEST(DataSetBoundaryTest, RejectsMissingDatasetPath) {
  const auto missingPath = testutils::namedFixturePath(
      "missing_" +
      testutils::sanitizePathComponent(
          ::testing::UnitTest::GetInstance()->current_test_info()->name()) +
      ".bin");
  testutils::ScopedPathCleanup cleanup(missingPath);

  EXPECT_THROW(
      {
        TypeParam dataSet(missingPath);
      },
      std::invalid_argument);
}

TEST(HDVectorBoundary, DistanceRejectsMismatchedDimensions) {
  HDVector left(std::vector<float>{1.0f, 2.0f});
  HDVector right(std::vector<float>{1.0f, 2.0f, 3.0f});

  EXPECT_THROW((void)euclideanDistance(left.view(), right.view()), std::invalid_argument);
}

TEST(GraphMutation, AddOutNeighbourUniqueSkipsDuplicatesAndClearRemovesEdges) {
  std::srand(0);
  Graph graph(3, 2);

  graph.clearOutNeighbors(0);
  graph.addOutNeighborUnique(0, 1);
  graph.addOutNeighborUnique(0, 1);
  graph.addOutNeighborUnique(0, 2);

  EXPECT_EQ(graph.getOutNeighbors(0), NodeList({1, 2}));

  graph.clearOutNeighbors(0);
  EXPECT_TRUE(graph.getOutNeighbors(0).empty());
}

TEST(GraphInitialization,
     UsesRequestedDegreeWhenEnoughUniqueNeighboursExistInDenseGraphs) {
  std::srand(0);
  Graph graph(64, 63);
  NodeList shortfallNodes;
  std::vector<uint64_t> observedDegrees;

  for (NodeId node = 0; node < 64; ++node) {
    const auto &neighbours = graph.getOutNeighbors(node);
    std::unordered_set<NodeId> unique(neighbours.begin(), neighbours.end());

    EXPECT_EQ(unique.size(), neighbours.size());
    EXPECT_EQ(unique.count(node), 0U);
    observedDegrees.push_back(static_cast<uint64_t>(neighbours.size()));
    if (neighbours.size() != 63U) {
      shortfallNodes.push_back(node);
    }
  }

  EXPECT_TRUE(shortfallNodes.empty())
      << "expected every node to have degree 63, but nodes "
      << ::testing::PrintToString(shortfallNodes)
      << " had degrees " << ::testing::PrintToString(observedDegrees);
}

TEST(GraphPersistence, LoadsSavedAdjacencyWithoutCrashing) {
  const auto path = testutils::namedFixturePath("graph_persistence_fixture.bin");
  testutils::ScopedPathCleanup cleanup(path);
  testutils::writeGraphFile(path, 3, 2, {{1, 2}, {0, 2}, {0, 1}});
  Graph graph(path);
  ASSERT_EQ(graph.getDegreeThreshold(), 2U);
  EXPECT_FALSE(graph.getMedoid().has_value());
  EXPECT_EQ(graph.getOutNeighbors(0), NodeList({1, 2}));
  EXPECT_EQ(graph.getOutNeighbors(1), NodeList({0, 2}));
  EXPECT_EQ(graph.getOutNeighbors(2), NodeList({0, 1}));
}

TEST(GraphPersistence, LoadsLargeSavedAdjacencyWithoutCrashing) {
  constexpr uint64_t kNodeCount = 64;
  constexpr uint64_t kDegreeThreshold = 6;
  const auto path =
      testutils::namedFixturePath("graph_persistence_large_fixture.bin");
  testutils::ScopedPathCleanup cleanup(path);
  testutils::writeGraphFile(
      path, kNodeCount, kDegreeThreshold,
      testutils::makeCircularAdjacency(kNodeCount, kDegreeThreshold));
  Graph graph(path);
  ASSERT_EQ(graph.getDegreeThreshold(),
            static_cast<uint64_t>(kDegreeThreshold));

  for (NodeId node = 0; node < kNodeCount; ++node) {
    const auto &neighbours = graph.getOutNeighbors(node);
    ASSERT_EQ(neighbours.size(), static_cast<size_t>(kDegreeThreshold));

    for (NodeId offset = 1; offset <= kDegreeThreshold; ++offset) {
      EXPECT_EQ(neighbours[offset - 1], (node + offset) % kNodeCount);
    }
  }
}

TEST(GraphPersistence, PreservesRandomizedVariableDegreeAdjacency) {
  constexpr uint64_t kNodeCount = 64;
  constexpr uint64_t kDegreeThreshold = 12;
  const auto path =
      testutils::uniqueFixturePath("boundary", "graph_variable_degree");
  testutils::ScopedPathCleanup cleanup(path);

  std::mt19937 rng(0x5EED1234u);
  std::uniform_int_distribution<uint64_t> degree_dist(1, kDegreeThreshold);
  std::vector<NodeList> expected(static_cast<size_t>(kNodeCount));
  for (NodeId node = 0; node < kNodeCount; ++node) {
    NodeList candidates;
    candidates.reserve(kNodeCount - 1);
    for (NodeId candidate = 0; candidate < kNodeCount; ++candidate) {
      if (candidate != node) {
        candidates.push_back(candidate);
      }
    }
    std::shuffle(candidates.begin(), candidates.end(), rng);

    const uint64_t degree = degree_dist(rng);
    expected[static_cast<size_t>(node)].assign(
        candidates.begin(),
        candidates.begin() + static_cast<std::ptrdiff_t>(degree));
  }

  testutils::writeGraphFile(path, kNodeCount, kDegreeThreshold, expected);

  Graph graph(path);
  ASSERT_EQ(graph.getDegreeThreshold(), kDegreeThreshold);
  for (NodeId node = 0; node < kNodeCount; ++node) {
    EXPECT_EQ(graph.getOutNeighbors(node), expected[static_cast<size_t>(node)])
        << "loaded adjacency changed for node " << node;
  }
}

TEST(GraphPersistence, RejectsAdjacencyDegreeAboveThreshold) {
  const auto path =
      testutils::uniqueFixturePath("boundary", "graph_degree_above_threshold");
  testutils::ScopedPathCleanup cleanup(path);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  const uint64_t node_count = 3;
  const uint64_t degree_threshold = 2;
  const uint64_t no_medoid = std::numeric_limits<uint64_t>::max();
  out.write(reinterpret_cast<const char *>(&node_count), sizeof(node_count));
  out.write(reinterpret_cast<const char *>(&degree_threshold),
            sizeof(degree_threshold));
  out.write(reinterpret_cast<const char *>(&no_medoid), sizeof(no_medoid));
  const uint64_t stored_degree = 3;
  const NodeList neighbours = {0, 1, 2};
  for (uint64_t node = 0; node < node_count; ++node) {
    out.write(reinterpret_cast<const char *>(&stored_degree),
              sizeof(stored_degree));
    out.write(reinterpret_cast<const char *>(neighbours.data()),
              static_cast<std::streamsize>(neighbours.size() *
                                           sizeof(NodeId)));
  }
  out.close();
  ASSERT_TRUE(out.good());

  EXPECT_THROW((void)Graph(path), std::runtime_error);
}

TEST(GraphPersistence, RejectsAdjacencyWithOutOfRangeNeighbourId) {
  const auto path =
      testutils::uniqueFixturePath("boundary", "graph_invalid_neighbour");
  testutils::ScopedPathCleanup cleanup(path);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  const uint64_t node_count = 3;
  const uint64_t degree_threshold = 2;
  const uint64_t no_medoid = std::numeric_limits<uint64_t>::max();
  out.write(reinterpret_cast<const char *>(&node_count), sizeof(node_count));
  out.write(reinterpret_cast<const char *>(&degree_threshold),
            sizeof(degree_threshold));
  out.write(reinterpret_cast<const char *>(&no_medoid), sizeof(no_medoid));

  const uint64_t stored_degree = 2;
  const std::vector<NodeList> adjacency = {{1, 2}, {0, 3}, {0, 1}};
  for (const auto &neighbours : adjacency) {
    out.write(reinterpret_cast<const char *>(&stored_degree),
              sizeof(stored_degree));
    out.write(reinterpret_cast<const char *>(neighbours.data()),
              static_cast<std::streamsize>(neighbours.size() *
                                           sizeof(NodeId)));
  }
  out.close();
  ASSERT_TRUE(out.good());

  EXPECT_THROW((void)Graph(path), std::runtime_error);
}

TEST(GraphPersistence, SaveRejectsAdjacencyDegreeAboveThreshold) {
  const auto path =
      testutils::uniqueFixturePath("boundary", "graph_save_degree_overflow");
  testutils::ScopedPathCleanup cleanup(path);

  Graph graph(4, 2);
  EXPECT_THROW(graph.setOutNeighbors(0, {1, 2, 3}), std::invalid_argument);
}

} // namespace
