#include "HDVector.hpp"
#include "dataset.hpp"
#include "graph.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace {

std::string sanitizePathComponent(std::string value) {
  for (char &ch : value) {
    const bool is_alnum = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                          (ch >= '0' && ch <= '9');
    if (!is_alnum && ch != '_' && ch != '-') {
      ch = '_';
    }
  }
  return value;
}

struct DatasetFixtureData {
  long long n = 4;
  long long dimensions = 3;
  std::vector<std::vector<float>> rows = {
      {10.0f, 1.0f, 2.0f},
      {20.0f, 3.0f, 4.0f},
      {30.0f, 5.0f, 6.0f},
      {40.0f, 7.0f, 8.0f},
  };
};

struct ScopedPathCleanup {
  explicit ScopedPathCleanup(std::filesystem::path path) : path(std::move(path)) {}

  ~ScopedPathCleanup() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  std::filesystem::path path;
};

std::filesystem::path makeDatasetFile(const std::string &name,
                                      const DatasetFixtureData &fixture) {
  const auto fixture_dir =
      std::filesystem::current_path() / "build" / "test-fixtures";
  std::filesystem::create_directories(fixture_dir);
  const auto path = fixture_dir / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("failed to create dataset fixture");
  }

  out.write(reinterpret_cast<const char *>(&fixture.n), sizeof(fixture.n));
  out.write(reinterpret_cast<const char *>(&fixture.dimensions),
            sizeof(fixture.dimensions));
  for (const auto &row : fixture.rows) {
    out.write(reinterpret_cast<const char *>(row.data()),
              static_cast<std::streamsize>(row.size() * sizeof(float)));
  }

  return path;
}

std::filesystem::path
makeGraphFile(const std::string &name, int nodeCount, int degreeThreshold,
              const std::vector<std::vector<int>> &adjacency) {
  const auto fixture_dir =
      std::filesystem::current_path() / "build" / "test-fixtures";
  std::filesystem::create_directories(fixture_dir);
  const auto path = fixture_dir / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("failed to create graph fixture");
  }

  out.write(reinterpret_cast<const char *>(&nodeCount), sizeof(nodeCount));
  out.write(reinterpret_cast<const char *>(&degreeThreshold),
            sizeof(degreeThreshold));
  for (const auto &neighbours : adjacency) {
    if (static_cast<int>(neighbours.size()) != degreeThreshold) {
      throw std::invalid_argument(
          "graph fixture adjacency does not match degree threshold");
    }
    out.write(reinterpret_cast<const char *>(neighbours.data()),
              static_cast<std::streamsize>(neighbours.size() * sizeof(int)));
  }

  return path;
}

std::vector<std::vector<int>> makeCircularAdjacency(int nodeCount,
                                                    int degreeThreshold) {
  std::vector<std::vector<int>> adjacency(nodeCount);
  for (int node = 0; node < nodeCount; ++node) {
    adjacency[node].reserve(degreeThreshold);
    for (int offset = 1; offset <= degreeThreshold; ++offset) {
      adjacency[node].push_back((node + offset) % nodeCount);
    }
  }
  return adjacency;
}

template <typename DataSetType>
class DataSetEdgeCaseTest : public ::testing::Test {
protected:
  DatasetFixtureData fixture;
  std::filesystem::path datasetPath;

  void SetUp() override {
    const auto *suite_name = ::testing::UnitTest::GetInstance()
                                 ->current_test_info()
                                 ->test_suite_name();
    const auto *test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    datasetPath = makeDatasetFile(
        std::string("edge_") + sanitizePathComponent(suite_name) + "_" +
            sanitizePathComponent(test_name) + ".bin",
        fixture);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove(datasetPath, ec);
  }

  std::unique_ptr<DataSetType> makeDataSet() {
    return std::make_unique<DataSetType>(datasetPath);
  }
};

using DataSetImplementations = ::testing::Types<FileDataSet, InMemoryDataSet>;
TYPED_TEST_SUITE(DataSetEdgeCaseTest, DataSetImplementations);

TYPED_TEST(DataSetEdgeCaseTest, AllowsEmptyRangesAtDatasetBoundary) {
  auto dataSet = this->makeDataSet();

  auto records = dataSet->getNRecordViewsFromIndex(this->fixture.rows.size(), 0);
  ASSERT_NE(records, nullptr);
  EXPECT_TRUE(records->empty());

  auto vectors = dataSet->getNHDVectorsFromIndex(this->fixture.rows.size(), 0);
  ASSERT_NE(vectors, nullptr);
  EXPECT_TRUE(vectors->empty());
}

TYPED_TEST(DataSetEdgeCaseTest, RejectsMissingDatasetPath) {
  const auto missingPath =
      std::filesystem::current_path() / "build" / "test-fixtures" /
      ("missing_" + sanitizePathComponent(
                        ::testing::UnitTest::GetInstance()
                            ->current_test_info()
                            ->name()) +
       ".bin");
  std::error_code ec;
  std::filesystem::remove(missingPath, ec);

  EXPECT_THROW(
      {
        TypeParam dataSet(missingPath);
      },
      std::invalid_argument);
}

TEST(HDVectorEdgeCases, DistanceRejectsMismatchedDimensions) {
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

  EXPECT_EQ(graph.getOutNeighbours(0), std::vector<int>({1, 2}));

  graph.clearOutNeighbours(0);
  EXPECT_TRUE(graph.getOutNeighbours(0).empty());
}

TEST(GraphInitialization,
     UsesRequestedDegreeWhenEnoughUniqueNeighboursExistInDenseGraphs) {
  std::srand(0);
  Graph graph(64, 63);
  std::vector<int> shortfallNodes;
  std::vector<int> observedDegrees;

  for (int node = 0; node < 64; ++node) {
    const auto &neighbours = graph.getOutNeighbours(node);
    std::unordered_set<int> unique(neighbours.begin(), neighbours.end());

    EXPECT_EQ(unique.size(), neighbours.size());
    EXPECT_EQ(unique.count(node), 0U);
    observedDegrees.push_back(static_cast<int>(neighbours.size()));
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
  const auto path = makeGraphFile("graph_persistence_fixture.bin", 3, 2,
                                  {{1, 2}, {0, 2}, {0, 1}});
  ScopedPathCleanup cleanup(path);
  const std::string graphPath = path.string();

  // This should succeed for a valid serialized graph fixture. It currently
  // fails because the graph-loading constructor never resizes its adjacency
  // vectors before reading into them.
  EXPECT_EXIT(
      {
        Graph graph(graphPath);
        if (graph.getDegreeThreshold() != 2) {
          std::exit(1);
        }

        const auto &nodeZero = graph.getOutNeighbours(0);
        if (nodeZero.size() != 2 || nodeZero[0] != 1 || nodeZero[1] != 2) {
          std::exit(2);
        }

        const auto &nodeOne = graph.getOutNeighbours(1);
        if (nodeOne.size() != 2 || nodeOne[0] != 0 || nodeOne[1] != 2) {
          std::exit(3);
        }

        const auto &nodeTwo = graph.getOutNeighbours(2);
        if (nodeTwo.size() != 2 || nodeTwo[0] != 0 || nodeTwo[1] != 1) {
          std::exit(4);
        }

        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(GraphPersistence, LoadsLargeSavedAdjacencyWithoutCrashing) {
  constexpr int kNodeCount = 64;
  constexpr int kDegreeThreshold = 6;
  const auto path =
      makeGraphFile("graph_persistence_large_fixture.bin", kNodeCount,
                    kDegreeThreshold,
                    makeCircularAdjacency(kNodeCount, kDegreeThreshold));
  ScopedPathCleanup cleanup(path);
  const std::string graphPath = path.string();

  EXPECT_EXIT(
      {
        Graph graph(graphPath);
        if (graph.getDegreeThreshold() != kDegreeThreshold) {
          std::exit(1);
        }

        for (int node = 0; node < kNodeCount; ++node) {
          const auto &neighbours = graph.getOutNeighbours(node);
          if (neighbours.size() != kDegreeThreshold) {
            std::exit(2);
          }

          for (int offset = 1; offset <= kDegreeThreshold; ++offset) {
            if (neighbours[offset - 1] != (node + offset) % kNodeCount) {
              std::exit(3);
            }
          }
        }

        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

} // namespace
