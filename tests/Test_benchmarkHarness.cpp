#include "benchmark_harness.hpp"
#include "test_utils.hpp"

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

std::filesystem::path writeBenchmarkFixture(const std::string &name) {
  const auto path = testutils::uniqueFixturePath("benchmark", name);
  testutils::writeDatasetFile(
      path, 5, 3,
      {{0.0f, 0.0f, 0.0f},
       {1.0f, 1.0f, 0.0f},
       {2.0f, 2.0f, 0.0f},
       {3.0f, 9.0f, 9.0f},
       {4.0f, 10.0f, 9.0f}});
  return path;
}

std::filesystem::path benchmarkArtifactDir(const std::string &name) {
  const auto dir = testutils::fixtureDir() / ("benchmark_artifacts_" + name);
  std::filesystem::create_directories(dir);
  return dir;
}

void removePathRecursively(const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::remove_all(path, ec);
}

}  // namespace

TEST(BenchmarkHarnessMath, ComputeRecallAtKIgnoresDuplicatesAndOrder) {
  const NodeList approximate = {2, 2, 3, 5, 7};
  const NodeList exact = {2, 3, 4};

  EXPECT_DOUBLE_EQ(computeRecallAtK(approximate, exact, 3), 2.0 / 3.0);
}

TEST(BenchmarkHarnessMath, ComputePercentileInterpolates) {
  EXPECT_DOUBLE_EQ(computePercentile({1.0, 2.0, 3.0, 4.0}, 0.50), 2.5);
  EXPECT_DOUBLE_EQ(computePercentile({1.0, 2.0, 3.0, 4.0}, 0.95), 3.85);
}

TEST(BenchmarkHarnessMath, FilterAndTruncateRemovesExcludedAndDuplicateIds) {
  const NodeList filtered =
      filterAndTruncateResults({1, 2, 2, 3, 4}, 3, OptionalNodeId(2));

  EXPECT_EQ(filtered, (NodeList{1, 3, 4}));
}

TEST(BenchmarkHarnessRuntime, BruteForceBenchmarkReportsPerfectRecall) {
  const auto datasetPath = writeBenchmarkFixture("dataset");
  const auto artifactDir = benchmarkArtifactDir("bruteforce");
  testutils::ScopedPathCleanup datasetCleanup(datasetPath);

  BenchmarkParameters parameters;
  parameters.algorithm = BenchmarkAlgorithm::BruteForce;
  parameters.datasetPath = datasetPath;
  parameters.datasetMode = BenchmarkDataSetMode::Memory;
  parameters.artifactDir = artifactDir;
  parameters.queryCount = 3;
  parameters.k = 2;
  parameters.seed = 1234;
  parameters.excludeSelf = false;

  const BenchmarkResult result = runBenchmark(parameters);
  const std::string json = benchmarkResultToJson(result);

  EXPECT_EQ(result.algorithm, BenchmarkAlgorithm::BruteForce);
  EXPECT_EQ(result.queryCount, 3U);
  EXPECT_DOUBLE_EQ(result.metrics.recallAtK, 1.0);
  EXPECT_FALSE(result.metrics.buildTimeSeconds.has_value());
  EXPECT_FALSE(result.metrics.ssdFootprintBytes.has_value());
  EXPECT_FALSE(result.metrics.restartTimeSeconds.has_value());

  const auto parsed = nlohmann::json::parse(json);
  EXPECT_EQ(parsed.at("algorithm"), "bruteforce");
  EXPECT_EQ(parsed.at("dataset").at("mode"), "memory");
  EXPECT_EQ(parsed.at("workload").at("query_dataset_path"), nullptr);
  EXPECT_EQ(parsed.at("parameters").at("degree_threshold"), nullptr);
  EXPECT_EQ(parsed.at("parameters").at("search_list_size"), nullptr);
  EXPECT_EQ(parsed.at("parameters").at("distance_threshold"), nullptr);
  EXPECT_EQ(parsed.at("metrics").at("build_time_seconds"), nullptr);
  EXPECT_EQ(parsed.at("metrics").at("ssd_footprint_bytes"), nullptr);
  EXPECT_EQ(parsed.at("metrics").at("restart_time_seconds"), nullptr);
  EXPECT_EQ(parsed.at("metrics").at("insert_throughput_vectors_per_second"),
            nullptr);

  removePathRecursively(artifactDir);
}

TEST(BenchmarkHarnessRuntime, VamanaBenchmarkReportsPersistenceMetrics) {
  const auto datasetPath = writeBenchmarkFixture("dataset");
  const auto artifactDir = benchmarkArtifactDir("vamana");
  testutils::ScopedPathCleanup datasetCleanup(datasetPath);

  BenchmarkParameters parameters;
  parameters.algorithm = BenchmarkAlgorithm::Vamana;
  parameters.datasetPath = datasetPath;
  parameters.datasetMode = BenchmarkDataSetMode::Memory;
  parameters.artifactDir = artifactDir;
  parameters.queryCount = 3;
  parameters.k = 1;
  parameters.seed = 9876;
  parameters.excludeSelf = false;
  parameters.degreeThreshold = 2;
  parameters.searchListSize = 4;
  parameters.distanceThreshold = 1.2f;

  const BenchmarkResult result = runBenchmark(parameters);
  const auto parsed = nlohmann::json::parse(benchmarkResultToJson(result));

  EXPECT_EQ(result.algorithm, BenchmarkAlgorithm::Vamana);
  EXPECT_EQ(result.queryCount, 3U);
  EXPECT_TRUE(result.metrics.buildTimeSeconds.has_value());
  EXPECT_TRUE(result.metrics.ssdFootprintBytes.has_value());
  EXPECT_GT(*result.metrics.ssdFootprintBytes, 0U);
  EXPECT_TRUE(result.metrics.restartTimeSeconds.has_value());
  EXPECT_TRUE(result.metrics.averageVisitedNodes.has_value());
  EXPECT_GE(result.metrics.recallAtK, 0.0);
  EXPECT_LE(result.metrics.recallAtK, 1.0);
  EXPECT_EQ(parsed.at("algorithm"), "vamana");
  EXPECT_EQ(parsed.at("dataset").at("mode"), "memory");
  EXPECT_EQ(parsed.at("parameters").at("degree_threshold"), 2);
  EXPECT_EQ(parsed.at("parameters").at("search_list_size"), 4);
  EXPECT_NEAR(parsed.at("parameters").at("distance_threshold").get<double>(),
              1.2, 1e-6);
  EXPECT_TRUE(parsed.at("metrics").at("build_time_seconds").is_number());
  EXPECT_TRUE(parsed.at("metrics").at("ssd_footprint_bytes").is_number());
  EXPECT_TRUE(parsed.at("metrics").at("restart_time_seconds").is_number());
  EXPECT_TRUE(parsed.at("metrics").at("average_visited_nodes").is_number());
  ASSERT_TRUE(result.notes.has_value());
  EXPECT_NE(result.notes->find("insert API"), std::string::npos);

  removePathRecursively(artifactDir);
}
