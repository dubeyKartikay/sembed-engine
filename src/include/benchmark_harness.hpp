#ifndef BENCHMARK_HARNESS
#define BENCHMARK_HARNESS

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "node_types.hpp"

enum class BenchmarkAlgorithm {
  BruteForce,
  Vamana,
};

struct BenchmarkParameters {
  BenchmarkAlgorithm algorithm = BenchmarkAlgorithm::BruteForce;
  std::filesystem::path datasetPath;
  std::optional<std::filesystem::path> queryDatasetPath;
  std::filesystem::path artifactDir = "benchmark-artifacts";
  uint64_t queryCount = 0;
  uint64_t k = 10;
  uint64_t seed = 0x53454d424544ULL;
  bool excludeSelf = true;
  uint64_t degreeThreshold = 64;
  uint64_t searchListSize = 100;
  float distanceThreshold = 1.2f;
};

struct BenchmarkMetrics {
  double recallAtK = 0.0;
  double latencyP50Ms = 0.0;
  double latencyP95Ms = 0.0;
  double queriesPerSecond = 0.0;
  std::optional<double> buildTimeSeconds;
  std::optional<uint64_t> ramFootprintBytes;
  std::optional<uint64_t> ssdFootprintBytes;
  std::optional<double> restartTimeSeconds;
  std::optional<double> insertThroughputVectorsPerSecond;
  std::optional<double> datasetLoadTimeSeconds;
  std::optional<double> queryDatasetLoadTimeSeconds;
  std::optional<double> averageVisitedNodes;
};

struct BenchmarkResult {
  BenchmarkAlgorithm algorithm = BenchmarkAlgorithm::BruteForce;
  std::filesystem::path datasetPath;
  std::optional<std::filesystem::path> queryDatasetPath;
  uint64_t datasetSize = 0;
  uint64_t dimensions = 0;
  uint64_t queryCount = 0;
  uint64_t k = 0;
  uint64_t seed = 0;
  bool excludeSelf = true;
  BenchmarkMetrics metrics;
  std::optional<uint64_t> degreeThreshold;
  std::optional<uint64_t> searchListSize;
  std::optional<float> distanceThreshold;
  std::optional<std::string> notes;
};

double computeRecallAtK(const NodeList &approximate, const NodeList &exact,
                        uint64_t k);
double computePercentile(std::vector<double> values, double percentile);
NodeList filterAndTruncateResults(const NodeList &input, uint64_t k,
                                  OptionalNodeId excludedNode = std::nullopt);

BenchmarkResult runBenchmark(const BenchmarkParameters &parameters);
std::string benchmarkResultToJson(const BenchmarkResult &result);

#endif  // BENCHMARK_HARNESS
