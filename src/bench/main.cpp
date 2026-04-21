#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "benchmark_harness.hpp"

namespace {

void printUsage(std::ostream &out) {
  out << "Usage: sembed_benchmark --algorithm <bruteforce|vamana> "
         "--dataset <path> [options]\n"
      << "Options:\n"
      << "  --query-dataset <path>        Optional query dataset.\n"
      << "  --dataset-mode <file|memory>  Dataset loading mode. Default: memory.\n"
      << "  --query-count <n>             Number of sampled queries. Default: min(100, N).\n"
      << "  --k <n>                       Recall/search depth. Default: 10.\n"
      << "  --seed <n>                    Deterministic workload seed.\n"
      << "  --exclude-self <true|false>   Exclude query id when queries come from the base dataset.\n"
      << "  --artifact-dir <path>         Directory for saved benchmark artifacts.\n"
      << "  --degree-threshold <n>        Vamana R parameter.\n"
      << "  --search-list-size <n>        Vamana L parameter.\n"
      << "  --distance-threshold <float>  Vamana alpha parameter.\n"
      << "  --output <path>               Write JSON output to a file instead of stdout.\n";
}

uint64_t parseUint64(const std::string &value, const std::string &flagName) {
  size_t consumed = 0;
  const uint64_t parsed = std::stoull(value, &consumed);
  if (consumed != value.size()) {
    throw std::invalid_argument("invalid integer for " + flagName + ": " + value);
  }
  return parsed;
}

float parseFloat(const std::string &value, const std::string &flagName) {
  size_t consumed = 0;
  const float parsed = std::stof(value, &consumed);
  if (consumed != value.size()) {
    throw std::invalid_argument("invalid float for " + flagName + ": " + value);
  }
  return parsed;
}

bool parseBool(const std::string &value, const std::string &flagName) {
  if (value == "true") {
    return true;
  }
  if (value == "false") {
    return false;
  }
  throw std::invalid_argument("invalid boolean for " + flagName + ": " + value);
}

std::string requireValue(int argc, char **argv, int &index,
                         const std::string &flagName) {
  if (index + 1 >= argc) {
    throw std::invalid_argument("missing value for " + flagName);
  }
  ++index;
  return argv[index];
}

}  // namespace

int main(int argc, char **argv) {
  try {
    BenchmarkParameters parameters;
    std::optional<std::filesystem::path> outputPath;
    bool algorithmSeen = false;
    bool datasetSeen = false;

    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (flag == "--help" || flag == "-h") {
        printUsage(std::cout);
        return 0;
      }
      if (flag == "--algorithm") {
        parameters.algorithm =
            parseBenchmarkAlgorithm(requireValue(argc, argv, i, flag));
        algorithmSeen = true;
        continue;
      }
      if (flag == "--dataset") {
        parameters.datasetPath = requireValue(argc, argv, i, flag);
        datasetSeen = true;
        continue;
      }
      if (flag == "--query-dataset") {
        parameters.queryDatasetPath = requireValue(argc, argv, i, flag);
        continue;
      }
      if (flag == "--dataset-mode") {
        parameters.datasetMode =
            parseBenchmarkDataSetMode(requireValue(argc, argv, i, flag));
        continue;
      }
      if (flag == "--query-count") {
        parameters.queryCount =
            parseUint64(requireValue(argc, argv, i, flag), flag);
        continue;
      }
      if (flag == "--k") {
        parameters.k = parseUint64(requireValue(argc, argv, i, flag), flag);
        continue;
      }
      if (flag == "--seed") {
        parameters.seed = parseUint64(requireValue(argc, argv, i, flag), flag);
        continue;
      }
      if (flag == "--exclude-self") {
        parameters.excludeSelf =
            parseBool(requireValue(argc, argv, i, flag), flag);
        continue;
      }
      if (flag == "--artifact-dir") {
        parameters.artifactDir = requireValue(argc, argv, i, flag);
        continue;
      }
      if (flag == "--degree-threshold") {
        parameters.degreeThreshold =
            parseUint64(requireValue(argc, argv, i, flag), flag);
        continue;
      }
      if (flag == "--search-list-size") {
        parameters.searchListSize =
            parseUint64(requireValue(argc, argv, i, flag), flag);
        continue;
      }
      if (flag == "--distance-threshold") {
        parameters.distanceThreshold =
            parseFloat(requireValue(argc, argv, i, flag), flag);
        continue;
      }
      if (flag == "--output") {
        outputPath = requireValue(argc, argv, i, flag);
        continue;
      }

      throw std::invalid_argument("unknown argument: " + flag);
    }

    if (!algorithmSeen || !datasetSeen) {
      printUsage(std::cerr);
      return 1;
    }

    const BenchmarkResult result = runBenchmark(parameters);
    const std::string json = benchmarkResultToJson(result);
    if (outputPath) {
      const std::filesystem::path parent = outputPath->parent_path();
      if (!parent.empty()) {
        std::filesystem::create_directories(parent);
      }
      std::ofstream out(*outputPath, std::ios::trunc);
      if (!out.is_open()) {
        throw std::runtime_error("failed to open output path for writing");
      }
      out << json << '\n';
      return 0;
    }

    std::cout << json << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "sembed_benchmark: " << error.what() << '\n';
    return 1;
  }
}
