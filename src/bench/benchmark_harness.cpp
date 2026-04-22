#include "benchmark_harness.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "HDVector.hpp"
#include "dataset.hpp"
#include "searchresults.hpp"
#include "utils.hpp"
#include "vamana.hpp"

NLOHMANN_JSON_SERIALIZE_ENUM(BenchmarkAlgorithm,
                             {{BenchmarkAlgorithm::BruteForce, "bruteforce"},
                              {BenchmarkAlgorithm::Vamana, "vamana"}})

NLOHMANN_JSON_SERIALIZE_ENUM(BenchmarkDataSetMode,
                             {{BenchmarkDataSetMode::File, "file"},
                              {BenchmarkDataSetMode::Memory, "memory"}})

namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

struct QueryWorkloadEntry {
  uint64_t queryIndex;
  HDVector vector;
  OptionalNodeId excludedBaseNode;
};

struct QueryMeasurement {
  NodeList resultIds;
  double latencyMs = 0.0;
  std::optional<double> visitedNodes;
};

double elapsedSeconds(const Clock::time_point &start,
                      const Clock::time_point &end) {
  return std::chrono::duration<double>(end - start).count();
}

double elapsedMilliseconds(const Clock::time_point &start,
                           const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string benchmarkAlgorithmName(BenchmarkAlgorithm algorithm) {
  switch (algorithm) {
    case BenchmarkAlgorithm::BruteForce:
      return "bruteforce";
    case BenchmarkAlgorithm::Vamana:
      return "vamana";
  }

  throw std::invalid_argument("unknown benchmark algorithm");
}

Json jsonPath(const std::filesystem::path &value) { return value.string(); }

template <typename T>
Json jsonOptional(const std::optional<T> &value) {
  return value ? Json(*value) : Json(nullptr);
}

Json jsonOptionalPath(const std::optional<std::filesystem::path> &value) {
  return value ? Json(value->string()) : Json(nullptr);
}

HDVector copyIntoHDVector(const Vector &source) {
  std::vector<float> values(static_cast<size_t>(source.getDimension()), 0.0f);
  for (uint64_t dim = 0; dim < source.getDimension(); ++dim) {
    values[static_cast<size_t>(dim)] = source[static_cast<int64_t>(dim)];
  }
  return HDVector(values);
}

float squaredDistance(const Vector &left, const Vector &right) {
  if (left.getDimension() != right.getDimension()) {
    throw std::invalid_argument("vector dimensions must match");
  }

  double total = 0.0;
  for (uint64_t dim = 0; dim < left.getDimension(); ++dim) {
    const double delta =
        static_cast<double>(left[static_cast<int64_t>(dim)]) -
        static_cast<double>(right[static_cast<int64_t>(dim)]);
    total += delta * delta;
  }
  return static_cast<float>(total);
}

std::unique_ptr<DataSet> makeDataSet(BenchmarkDataSetMode mode,
                                     const std::filesystem::path &path) {
  switch (mode) {
    case BenchmarkDataSetMode::File:
      return std::make_unique<FileDataSet>(path);
    case BenchmarkDataSetMode::Memory:
      return std::make_unique<InMemoryDataSet>(path);
  }

  throw std::invalid_argument("unsupported dataset mode");
}

NodeList exactNearestNeighbors(DataSet &baseDataSet, const Vector &queryVector,
                               uint64_t k, OptionalNodeId excludedBaseNode) {
  std::vector<std::pair<float, NodeId>> ranked;
  ranked.reserve(static_cast<size_t>(baseDataSet.getN()));

  for (NodeId node = 0; node < baseDataSet.getN(); ++node) {
    if (excludedBaseNode && node == *excludedBaseNode) {
      continue;
    }

    const RecordView record = baseDataSet.getRecordViewByIndex(node);
    ranked.push_back({squaredDistance(queryVector, *record.vector), node});
  }

  const size_t limit =
      std::min<size_t>(static_cast<size_t>(k), ranked.size());
  std::partial_sort(
      ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(limit),
      ranked.end(), [](const auto &left, const auto &right) {
        if (left.first == right.first) {
          return left.second < right.second;
        }
        return left.first < right.first;
      });

  NodeList exact;
  exact.reserve(limit);
  for (size_t i = 0; i < limit; ++i) {
    exact.push_back(ranked[i].second);
  }
  return exact;
}

std::vector<QueryWorkloadEntry> buildQueryWorkload(
    DataSet &baseDataSet, DataSet &queryDataSet, uint64_t requestedQueryCount,
    uint64_t seed, bool excludeSelf) {
  const uint64_t availableQueries = queryDataSet.getN();
  const uint64_t queryCount =
      requestedQueryCount == 0 ? std::min<uint64_t>(100, availableQueries)
                               : std::min<uint64_t>(requestedQueryCount,
                                                    availableQueries);

  std::vector<QueryWorkloadEntry> workload;
  workload.reserve(static_cast<size_t>(queryCount));
  if (queryCount == 0) {
    return workload;
  }

  auto rng = makeDeterministicRng(0x62656e63686d6172ULL,
                                  {availableQueries, queryCount, seed,
                                   baseDataSet.getN(), queryDataSet.getN()});
  const NodeList queryIndices =
      generateRandomNumbers(queryCount, availableQueries, rng);

  for (NodeId queryIndex : queryIndices) {
    const RecordView queryRecord = queryDataSet.getRecordViewByIndex(queryIndex);
    workload.push_back({queryIndex, copyIntoHDVector(*queryRecord.vector),
                        excludeSelf ? OptionalNodeId(queryIndex)
                                    : std::nullopt});
  }

  return workload;
}

std::vector<QueryMeasurement> runBruteForceQueries(
    DataSet &baseDataSet, const std::vector<QueryWorkloadEntry> &workload,
    uint64_t k) {
  std::vector<QueryMeasurement> measurements;
  measurements.reserve(workload.size());

  for (const QueryWorkloadEntry &query : workload) {
    const Clock::time_point start = Clock::now();
    NodeList exact =
        exactNearestNeighbors(baseDataSet, query.vector, k, query.excludedBaseNode);
    const Clock::time_point end = Clock::now();

    measurements.push_back(
        {std::move(exact), elapsedMilliseconds(start, end), std::nullopt});
  }

  return measurements;
}

std::vector<QueryMeasurement> runVamanaQueries(
    Vamana &index, const std::vector<QueryWorkloadEntry> &workload, uint64_t k) {
  std::vector<QueryMeasurement> measurements;
  measurements.reserve(workload.size());

  for (const QueryWorkloadEntry &query : workload) {
    const uint64_t requestedK = query.excludedBaseNode ? k + 1 : k;
    const Clock::time_point start = Clock::now();
    const SearchResults results = index.greedySearch(query.vector, requestedK);
    const Clock::time_point end = Clock::now();

    measurements.push_back(
        {filterAndTruncateResults(results.approximateNN, k, query.excludedBaseNode),
         elapsedMilliseconds(start, end),
         static_cast<double>(results.visited.size())});
  }

  return measurements;
}

std::vector<NodeList> buildExactResults(
    DataSet &baseDataSet, const std::vector<QueryWorkloadEntry> &workload,
    uint64_t k) {
  std::vector<NodeList> exactResults;
  exactResults.reserve(workload.size());

  for (const QueryWorkloadEntry &query : workload) {
    exactResults.push_back(
        exactNearestNeighbors(baseDataSet, query.vector, k, query.excludedBaseNode));
  }

  return exactResults;
}

double totalLatencyMilliseconds(
    const std::vector<QueryMeasurement> &measurements) {
  double total = 0.0;
  for (const QueryMeasurement &measurement : measurements) {
    total += measurement.latencyMs;
  }
  return total;
}

std::optional<double> meanVisitedNodes(
    const std::vector<QueryMeasurement> &measurements) {
  double total = 0.0;
  size_t counted = 0;
  for (const QueryMeasurement &measurement : measurements) {
    if (!measurement.visitedNodes) {
      continue;
    }
    total += *measurement.visitedNodes;
    ++counted;
  }

  if (counted == 0) {
    return std::nullopt;
  }
  return total / static_cast<double>(counted);
}

std::optional<uint64_t> currentResidentSetBytes() {
#if defined(__APPLE__)
  mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  const kern_return_t status =
      task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count);
  if (status != KERN_SUCCESS) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(info.resident_size);
#elif defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  if (!statm.is_open()) {
    return std::nullopt;
  }

  uint64_t totalPages = 0;
  uint64_t residentPages = 0;
  statm >> totalPages >> residentPages;
  if (!statm) {
    return std::nullopt;
  }

  const long pageSize = sysconf(_SC_PAGESIZE);
  if (pageSize <= 0) {
    return std::nullopt;
  }
  return residentPages * static_cast<uint64_t>(pageSize);
#else
  return std::nullopt;
#endif
}

std::optional<uint64_t> residentDeltaBytes(
    const std::optional<uint64_t> &before,
    const std::optional<uint64_t> &after) {
  if (!before || !after || *after < *before) {
    return std::nullopt;
  }
  return *after - *before;
}

std::filesystem::path makeArtifactPath(const std::filesystem::path &artifactDir,
                                       BenchmarkAlgorithm algorithm) {
  std::filesystem::create_directories(artifactDir);
  const auto stamp =
      std::chrono::duration_cast<std::chrono::microseconds>(
          Clock::now().time_since_epoch())
          .count();
  return artifactDir /
         (benchmarkAlgorithmName(algorithm) + "_" + std::to_string(stamp) +
          ".graph");
}

}  // namespace

double computeRecallAtK(const NodeList &approximate, const NodeList &exact,
                        uint64_t k) {
  if (k == 0) {
    return 1.0;
  }

  const NodeList approximateTopK = filterAndTruncateResults(approximate, k);
  const NodeList exactTopK = filterAndTruncateResults(exact, k);
  if (exactTopK.empty()) {
    return 1.0;
  }

  std::unordered_set<NodeId> truth(exactTopK.begin(), exactTopK.end());
  uint64_t hits = 0;
  for (NodeId node : approximateTopK) {
    const auto hit = truth.find(node);
    if (hit != truth.end()) {
      ++hits;
      truth.erase(hit);
    }
  }

  return static_cast<double>(hits) / static_cast<double>(exactTopK.size());
}

double computePercentile(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }

  if (percentile <= 0.0) {
    return *std::min_element(values.begin(), values.end());
  }
  if (percentile >= 1.0) {
    return *std::max_element(values.begin(), values.end());
  }

  std::sort(values.begin(), values.end());
  const double scaledIndex =
      percentile * static_cast<double>(values.size() - 1);
  const size_t lowerIndex = static_cast<size_t>(std::floor(scaledIndex));
  const size_t upperIndex = static_cast<size_t>(std::ceil(scaledIndex));
  if (lowerIndex == upperIndex) {
    return values[lowerIndex];
  }

  const double fraction = scaledIndex - static_cast<double>(lowerIndex);
  return values[lowerIndex] +
         (values[upperIndex] - values[lowerIndex]) * fraction;
}

NodeList filterAndTruncateResults(const NodeList &input, uint64_t k,
                                  OptionalNodeId excludedNode) {
  NodeList filtered;
  filtered.reserve(std::min<size_t>(input.size(), static_cast<size_t>(k)));
  std::unordered_set<NodeId> seen;

  for (NodeId node : input) {
    if (excludedNode && node == *excludedNode) {
      continue;
    }
    if (seen.insert(node).second) {
      filtered.push_back(node);
    }
    if (filtered.size() == static_cast<size_t>(k)) {
      break;
    }
  }

  return filtered;
}

void to_json(nlohmann::json &json, const BenchmarkMetrics &metrics) {
  json = {{"recall_at_k", metrics.recallAtK},
          {"latency_p50_ms", metrics.latencyP50Ms},
          {"latency_p95_ms", metrics.latencyP95Ms},
          {"queries_per_second", metrics.queriesPerSecond},
          {"build_time_seconds", jsonOptional(metrics.buildTimeSeconds)},
          {"ram_footprint_bytes", jsonOptional(metrics.ramFootprintBytes)},
          {"ssd_footprint_bytes", jsonOptional(metrics.ssdFootprintBytes)},
          {"restart_time_seconds", jsonOptional(metrics.restartTimeSeconds)},
          {"insert_throughput_vectors_per_second",
           jsonOptional(metrics.insertThroughputVectorsPerSecond)},
          {"dataset_load_time_seconds",
           jsonOptional(metrics.datasetLoadTimeSeconds)},
          {"query_dataset_load_time_seconds",
           jsonOptional(metrics.queryDatasetLoadTimeSeconds)},
          {"average_visited_nodes", jsonOptional(metrics.averageVisitedNodes)}};
}

void to_json(nlohmann::json &json, const BenchmarkResult &result) {
  json = {{"algorithm", result.algorithm},
          {"dataset",
           {{"path", jsonPath(result.datasetPath)},
            {"mode", result.datasetMode},
            {"records", result.datasetSize},
            {"dimensions", result.dimensions}}},
          {"workload",
           {{"query_dataset_path", jsonOptionalPath(result.queryDatasetPath)},
            {"query_count", result.queryCount},
            {"k", result.k},
            {"seed", result.seed},
            {"exclude_self", result.excludeSelf}}},
          {"parameters",
           {{"degree_threshold", jsonOptional(result.degreeThreshold)},
            {"search_list_size", jsonOptional(result.searchListSize)},
            {"distance_threshold", jsonOptional(result.distanceThreshold)}}},
          {"metrics", result.metrics},
          {"notes", jsonOptional(result.notes)}};
}

BenchmarkResult runBenchmark(const BenchmarkParameters &parameters) {
  if (parameters.k == 0) {
    throw std::invalid_argument("k must be greater than zero");
  }

  const std::optional<uint64_t> rssBeforeLoad = currentResidentSetBytes();

  const Clock::time_point baseLoadStart = Clock::now();
  std::unique_ptr<DataSet> baseDataSet =
      makeDataSet(parameters.datasetMode, parameters.datasetPath);
  const Clock::time_point baseLoadEnd = Clock::now();

  std::unique_ptr<DataSet> queryDataSet;
  std::optional<double> queryLoadTimeSeconds;
  if (parameters.queryDatasetPath) {
    const Clock::time_point queryLoadStart = Clock::now();
    queryDataSet =
        makeDataSet(parameters.datasetMode, *parameters.queryDatasetPath);
    const Clock::time_point queryLoadEnd = Clock::now();
    queryLoadTimeSeconds =
        elapsedSeconds(queryLoadStart, queryLoadEnd);
  }

  DataSet &querySource = queryDataSet ? *queryDataSet : *baseDataSet;
  const bool excludeSelf = !parameters.queryDatasetPath.has_value() &&
                           parameters.excludeSelf;

  std::vector<QueryWorkloadEntry> workload = buildQueryWorkload(
      *baseDataSet, querySource, parameters.queryCount, parameters.seed,
      excludeSelf);

  BenchmarkResult result;
  result.algorithm = parameters.algorithm;
  result.datasetMode = parameters.datasetMode;
  result.datasetPath = parameters.datasetPath;
  result.queryDatasetPath = parameters.queryDatasetPath;
  result.datasetSize = baseDataSet->getN();
  result.dimensions = baseDataSet->getDimensions();
  result.queryCount = static_cast<uint64_t>(workload.size());
  result.k = parameters.k;
  result.seed = parameters.seed;
  result.excludeSelf = excludeSelf;
  result.metrics.datasetLoadTimeSeconds =
      elapsedSeconds(baseLoadStart, baseLoadEnd);
  result.metrics.queryDatasetLoadTimeSeconds = queryLoadTimeSeconds;

  if (parameters.algorithm == BenchmarkAlgorithm::BruteForce) {
    const std::vector<QueryMeasurement> measurements =
        runBruteForceQueries(*baseDataSet, workload, parameters.k);

    std::vector<double> latenciesMs;
    latenciesMs.reserve(measurements.size());
    for (const QueryMeasurement &measurement : measurements) {
      latenciesMs.push_back(measurement.latencyMs);
    }

    result.metrics.recallAtK = 1.0;
    result.metrics.latencyP50Ms = computePercentile(latenciesMs, 0.50);
    result.metrics.latencyP95Ms = computePercentile(latenciesMs, 0.95);
    const double totalQueryMs = totalLatencyMilliseconds(measurements);
    result.metrics.queriesPerSecond =
        totalQueryMs == 0.0 ? 0.0
                            : (static_cast<double>(measurements.size()) * 1000.0) /
                                  totalQueryMs;
    result.metrics.ramFootprintBytes =
        residentDeltaBytes(rssBeforeLoad, currentResidentSetBytes());
    result.notes =
        "Insert throughput is unavailable because the current engine exposes "
        "no insert API yet.";
    return result;
  }

  std::vector<NodeList> exactResults =
      buildExactResults(*baseDataSet, workload, parameters.k);
  const Clock::time_point buildStart = Clock::now();
  Vamana index(std::move(baseDataSet), parameters.degreeThreshold,
               parameters.distanceThreshold);
  const Clock::time_point buildEnd = Clock::now();
  index.setSearchListSize(static_cast<int64_t>(parameters.searchListSize));

  const std::vector<QueryMeasurement> measurements =
      runVamanaQueries(index, workload, parameters.k);

  std::vector<double> latenciesMs;
  latenciesMs.reserve(measurements.size());
  double recallTotal = 0.0;
  for (size_t i = 0; i < measurements.size(); ++i) {
    latenciesMs.push_back(measurements[i].latencyMs);
    recallTotal += computeRecallAtK(measurements[i].resultIds, exactResults[i],
                                    parameters.k);
  }

  const std::filesystem::path artifactPath =
      makeArtifactPath(parameters.artifactDir, parameters.algorithm);
  index.save(artifactPath);

  const Clock::time_point restartStart = Clock::now();
  std::unique_ptr<DataSet> reloadedDataSet =
      makeDataSet(parameters.datasetMode, parameters.datasetPath);
  Vamana reloadedIndex(std::move(reloadedDataSet), artifactPath,
                       parameters.distanceThreshold);
  reloadedIndex.setSearchListSize(static_cast<int64_t>(parameters.searchListSize));
  const Clock::time_point restartEnd = Clock::now();

  result.degreeThreshold = parameters.degreeThreshold;
  result.searchListSize = parameters.searchListSize;
  result.distanceThreshold = parameters.distanceThreshold;
  result.metrics.recallAtK =
      measurements.empty() ? 1.0
                           : recallTotal / static_cast<double>(measurements.size());
  result.metrics.latencyP50Ms = computePercentile(latenciesMs, 0.50);
  result.metrics.latencyP95Ms = computePercentile(latenciesMs, 0.95);
  const double totalQueryMs = totalLatencyMilliseconds(measurements);
  result.metrics.queriesPerSecond =
      totalQueryMs == 0.0
          ? 0.0
          : (static_cast<double>(measurements.size()) * 1000.0) / totalQueryMs;
  result.metrics.buildTimeSeconds = elapsedSeconds(buildStart, buildEnd);
  result.metrics.ramFootprintBytes =
      residentDeltaBytes(rssBeforeLoad, currentResidentSetBytes());
  result.metrics.ssdFootprintBytes = std::filesystem::file_size(artifactPath);
  result.metrics.restartTimeSeconds =
      elapsedSeconds(restartStart, restartEnd);
  result.metrics.averageVisitedNodes = meanVisitedNodes(measurements);
  result.notes =
      "Insert throughput is unavailable because the current engine exposes no "
      "insert API yet.";
  return result;
}

std::string benchmarkResultToJson(const BenchmarkResult &result) {
  return nlohmann::json(result).dump(2);
}
