#include <algorithm>

#include "HDVector.hpp"
#include "test_utils.hpp"
#include "vamana.hpp"
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

namespace {

struct AnnFixtureData {
  int64_t n = 6;
  int64_t dimensions = 3;
  std::vector<std::vector<float>> rows = {
      {0.0f, 0.0f, 0.0f},
      {1.0f, 1.0f, 0.0f},
      {2.0f, 2.0f, 0.0f},
      {3.0f, 10.0f, 10.0f},
      {4.0f, 11.0f, 10.0f},
      {5.0f, 12.0f, 10.0f},
  };
};

AnnFixtureData makeLargeClusteredFixture() {
  AnnFixtureData fixture;
  fixture.dimensions = 3;
  fixture.rows.clear();

  constexpr uint64_t clusterCount = 4;
  constexpr uint64_t pointsPerCluster = 8;
  int64_t id = 0;
  for (uint64_t cluster = 0; cluster < clusterCount; ++cluster) {
    const float baseX = static_cast<float>(cluster * 100);
    const float baseY = (cluster % 2 == 0) ? 0.0f : 80.0f;
    for (uint64_t point = 0; point < pointsPerCluster; ++point) {
      const float x = baseX + static_cast<float>(point);
      const float y = baseY + static_cast<float>(point % 3) * 0.5f;
      fixture.rows.push_back({static_cast<float>(id), x, y});
      ++id;
    }
  }

  fixture.n = static_cast<int64_t>(fixture.rows.size());
  return fixture;
}

float squaredDistance(const std::vector<float> &left,
                      const std::vector<float> &right) {
  float total = 0.0f;
  for (int64_t i = 0; i < static_cast<int64_t>(left.size()); ++i) {
    const float delta = left[i] - right[i];
    total += delta * delta;
  }
  return total;
}

NodeList exactNearestIds(const AnnFixtureData &fixture,
                         const std::vector<float> &query, uint64_t k) {
  std::vector<std::pair<float, NodeId>> ranked;
  ranked.reserve(fixture.rows.size());
  for (const auto &row : fixture.rows) {
    const std::vector<float> payload(row.begin() + 1, row.end());
    ranked.push_back(
        {squaredDistance(payload, query), static_cast<NodeId>(row.front())});
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const auto &left, const auto &right) {
              if (left.first == right.first) {
                return left.second < right.second;
              }
              return left.first < right.first;
            });

  NodeList result;
  result.reserve(k);
  for (uint64_t i = 0; i < k && i < ranked.size(); ++i) {
    result.push_back(ranked[i].second);
  }
  return result;
}

template <typename DataSetType>
class DeterministicANNTest : public ::testing::Test {
protected:
  AnnFixtureData fixture;
  std::filesystem::path datasetPath;

  void SetUp() override {
    datasetPath = testutils::uniqueFixturePath("vamana", "dataset");
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

using DeterministicDataSets = ::testing::Types<FileDataSet, InMemoryDataSet>;
TYPED_TEST_SUITE(DeterministicANNTest, DeterministicDataSets);

template <typename DataSetType>
class LargeDeterministicANNTest : public ::testing::Test {
protected:
  AnnFixtureData fixture = makeLargeClusteredFixture();
  std::filesystem::path datasetPath;

  void SetUp() override {
    datasetPath = testutils::uniqueFixturePath("vamana", "dataset");
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

TYPED_TEST_SUITE(LargeDeterministicANNTest, DeterministicDataSets);

TYPED_TEST(DeterministicANNTest, BuildIndexKeepsBoundedUniqueNeighbours) {
  std::srand(0);
  auto dataSet = this->makeDataSet();
  Vamana vamana(std::move(dataSet), 2);

  for (NodeId node = 0; node < static_cast<NodeId>(this->fixture.n); ++node) {
    const NodeList &neighbours = vamana.getOutNeighbors(node);
    EXPECT_FALSE(neighbours.empty());
    EXPECT_LE(neighbours.size(), 2U);

    std::unordered_set<NodeId> unique;
    for (NodeId neighbour : neighbours) {
      EXPECT_NE(neighbour, node);
      unique.insert(neighbour);
    }
    EXPECT_EQ(unique.size(), neighbours.size());
  }
}

TEST(VamanaIndexConstruction, BuildIndexDoesNotDuplicateExistingBacklinks) {
  const AnnFixtureData fixture;
  const auto datasetPath =
      testutils::uniqueFixturePath("vamana_backlinks", "dataset");
  testutils::ScopedPathCleanup cleanup(datasetPath);
  testutils::writeDatasetFile(datasetPath, fixture.n, fixture.dimensions,
                              fixture.rows);

  std::srand(0);
  auto dataSet = std::make_unique<FileDataSet>(datasetPath);
  Vamana vamana(std::move(dataSet), 3);

  // Seed a reciprocal edge from a late node back to an early node in the fixed
  // permutation order so the duplicate backlink survives the build.
  vamana.setOutNeighbors(0, {2});
  vamana.setOutNeighbors(2, {0});
  for (NodeId node = 1; node < static_cast<NodeId>(fixture.n); ++node) {
    if (node == 2) {
      continue;
    }
    vamana.setOutNeighbors(node, {2});
  }

  vamana.buildIndex();

  for (NodeId node = 0; node < static_cast<NodeId>(fixture.n); ++node) {
    const auto &neighbours = vamana.getOutNeighbors(node);
    std::unordered_set<NodeId> uniqueNonSelfNeighbours;
    size_t nonSelfCount = 0;

    for (NodeId neighbour : neighbours) {
      if (neighbour == node) {
        continue;
      }
      ++nonSelfCount;
      uniqueNonSelfNeighbours.insert(neighbour);
    }

    EXPECT_EQ(uniqueNonSelfNeighbours.size(), nonSelfCount)
        << "node " << node << " contains duplicated non-self neighbours";
  }

  std::error_code ec;
  std::filesystem::remove(datasetPath, ec);
}

TYPED_TEST(DeterministicANNTest, SelfQueriesReturnExactRecord) {
  std::srand(0);
  auto dataSet = this->makeDataSet();
  Vamana vamana(std::move(dataSet), 3);
  vamana.setSearchListSize(this->fixture.n);

  for (NodeId row_index = 0;
       row_index < static_cast<NodeId>(this->fixture.rows.size());
       ++row_index) {
    const std::vector<float> queryValues(this->fixture.rows[row_index].begin() + 1,
                                         this->fixture.rows[row_index].end());
    HDVector query(queryValues);
    SearchResults results = vamana.greedySearch(query, 1);

    ASSERT_EQ(results.approximateNN.size(), 1U);
    EXPECT_EQ(results.approximateNN.front(), row_index);
  }
}

TYPED_TEST(DeterministicANNTest, GreedySearchReturnsCandidatesSortedByDistance) {
  std::srand(0);
  auto dataSet = this->makeDataSet();
  Vamana vamana(std::move(dataSet), 3);
  vamana.setSearchListSize(this->fixture.n);

  const std::vector<float> queryValues = {11.1f, 10.0f};
  HDVector query(queryValues);
  SearchResults results = vamana.greedySearch(query, 3);

  ASSERT_EQ(results.approximateNN.size(), 3U);

  std::unordered_set<NodeId> unique;
  float previousDistance = -1.0f;
  for (NodeId candidate : results.approximateNN) {
    unique.insert(candidate);

    const auto record = vamana.getRecordViewByIndex(candidate);
    std::vector<float> payload(record.vector->getDimension(), 0.0f);
    for (int64_t dim = 0; dim < static_cast<int64_t>(record.vector->getDimension()); ++dim) {
      payload[dim] = (*record.vector)[dim];
    }
    const float currentDistance = squaredDistance(payload, queryValues);
    EXPECT_GE(currentDistance, previousDistance);
    previousDistance = currentDistance;
  }
  EXPECT_EQ(unique.size(), results.approximateNN.size());
}

TYPED_TEST(DeterministicANNTest,
           GreedySearchMatchesExactNearestNeighborOnSeparatedQuery) {
  std::srand(0);
  auto dataSet = this->makeDataSet();
  Vamana vamana(std::move(dataSet), 3);
  vamana.setSearchListSize(this->fixture.n);

  const std::vector<float> queryValues = {11.1f, 10.0f};
  HDVector query(queryValues);
  SearchResults results = vamana.greedySearch(query, 1);
  const NodeList exact = exactNearestIds(this->fixture, queryValues, 1);

  ASSERT_EQ(results.approximateNN.size(), 1U);
  ASSERT_EQ(exact.size(), 1U);
  EXPECT_EQ(results.approximateNN.front(), exact.front());
}

TYPED_TEST(LargeDeterministicANNTest, BuildIndexKeepsBoundedUniqueNeighbours) {
  std::srand(0);
  auto dataSet = this->makeDataSet();
  Vamana vamana(std::move(dataSet), 5);

  for (NodeId node = 0; node < static_cast<NodeId>(this->fixture.n); ++node) {
    const NodeList &neighbours = vamana.getOutNeighbors(node);
    EXPECT_FALSE(neighbours.empty());
    EXPECT_LE(neighbours.size(), 5U);

    std::unordered_set<NodeId> unique;
    for (NodeId neighbour : neighbours) {
      EXPECT_NE(neighbour, node);
      unique.insert(neighbour);
    }
    EXPECT_EQ(unique.size(), neighbours.size());
  }
}

TYPED_TEST(LargeDeterministicANNTest, SelfQueriesReturnExactRecordAcrossGraph) {
  std::srand(0);
  auto dataSet = this->makeDataSet();
  Vamana vamana(std::move(dataSet), 6);
  vamana.setSearchListSize(this->fixture.n);

  for (NodeId row_index = 0;
       row_index < static_cast<NodeId>(this->fixture.rows.size());
       ++row_index) {
    const std::vector<float> queryValues(this->fixture.rows[row_index].begin() + 1,
                                         this->fixture.rows[row_index].end());
    HDVector query(queryValues);
    SearchResults results = vamana.greedySearch(query, 1);

    ASSERT_EQ(results.approximateNN.size(), 1U);
    EXPECT_EQ(results.approximateNN.front(), row_index);
  }
}

TYPED_TEST(LargeDeterministicANNTest,
           GreedySearchMatchesExactTopKOnSeparatedClusterQuery) {
  std::srand(0);
  auto dataSet = this->makeDataSet();
  Vamana vamana(std::move(dataSet), 6);
  vamana.setSearchListSize(this->fixture.n);

  const std::vector<float> queryValues = {203.25f, 0.5f};
  HDVector query(queryValues);
  SearchResults results = vamana.greedySearch(query, 5);
  const NodeList exact = exactNearestIds(this->fixture, queryValues, 5);

  ASSERT_EQ(results.approximateNN.size(), 5U);
  ASSERT_EQ(exact.size(), 5U);
  EXPECT_EQ(results.approximateNN, exact);
}

} // namespace
