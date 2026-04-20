// Boundary-condition and persistence regression tests for dataset and graph
// loading, mutation, and serialization behavior.

#include "HDVector.hpp"
#include "dataset.hpp"
#include "graph.hpp"
#include "test_utils.hpp"
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
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

using DataSetImplementations = ::testing::Types<FileDataSet, InMemoryDataSet>;
TYPED_TEST_SUITE(DataSetBoundaryTest, DataSetImplementations);

TYPED_TEST(DataSetBoundaryTest, AllowsEmptyRangesAtDatasetBoundary) {
  auto dataSet = this->makeDataSet();

  auto records = dataSet->getNRecordViewsFromIndex(this->fixture.rows.size(), 0);
  ASSERT_NE(records, nullptr);
  EXPECT_TRUE(records->empty());

  auto vectors = dataSet->getNHDVectorsFromIndex(this->fixture.rows.size(), 0);
  ASSERT_NE(vectors, nullptr);
  EXPECT_TRUE(vectors->empty());
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

  EXPECT_THROW((void)HDVector::distance(left, right), std::invalid_argument);
}

TEST(GraphMutation, AddOutNeighbourUniqueSkipsDuplicatesAndClearRemovesEdges) {
  std::srand(0);
  Graph graph(3, 2);

  graph.clearOutNeighbours(0);
  graph.addOutNeighbourUnique(0, 1);
  graph.addOutNeighbourUnique(0, 1);
  graph.addOutNeighbourUnique(0, 2);

  EXPECT_EQ(graph.getOutNeighbours(0), NodeList({1, 2}));

  graph.clearOutNeighbours(0);
  EXPECT_TRUE(graph.getOutNeighbours(0).empty());
}

TEST(GraphInitialization,
     UsesRequestedDegreeWhenEnoughUniqueNeighboursExistInDenseGraphs) {
  std::srand(0);
  Graph graph(64, 63);
  NodeList shortfallNodes;
  std::vector<uint64_t> observedDegrees;

  for (NodeId node = 0; node < 64; ++node) {
    const auto &neighbours = graph.getOutNeighbours(node);
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
  EXPECT_FALSE(graph.getMediod().has_value());
  EXPECT_EQ(graph.getOutNeighbours(0), NodeList({1, 2}));
  EXPECT_EQ(graph.getOutNeighbours(1), NodeList({0, 2}));
  EXPECT_EQ(graph.getOutNeighbours(2), NodeList({0, 1}));
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
    const auto &neighbours = graph.getOutNeighbours(node);
    ASSERT_EQ(neighbours.size(), static_cast<size_t>(kDegreeThreshold));

    for (NodeId offset = 1; offset <= kDegreeThreshold; ++offset) {
      EXPECT_EQ(neighbours[offset - 1], (node + offset) % kNodeCount);
    }
  }
}

} // namespace
