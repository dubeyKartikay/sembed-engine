#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <CLI/CLI.hpp>

#include "benchmark_harness.hpp"

namespace {

BenchmarkAlgorithm benchmarkAlgorithmFromName(const std::string &value) {
  if (value == "bruteforce" || value == "brute-force") {
    return BenchmarkAlgorithm::BruteForce;
  }
  if (value == "vamana") {
    return BenchmarkAlgorithm::Vamana;
  }

  throw std::invalid_argument("unknown benchmark algorithm: " + value);
}

BenchmarkDataSetMode benchmarkDataSetModeFromName(const std::string &value) {
  if (value == "file") {
    return BenchmarkDataSetMode::File;
  }
  if (value == "memory") {
    return BenchmarkDataSetMode::Memory;
  }

  throw std::invalid_argument("unknown dataset mode: " + value);
}

bool parseBoolValue(const std::string &value) {
  if (value == "true") {
    return true;
  }
  if (value == "false") {
    return false;
  }

  throw std::invalid_argument("unknown boolean value: " + value);
}

void writeBenchmarkResult(const std::string &json,
                          const std::filesystem::path &outputPath) {
  const std::filesystem::path parent = outputPath.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }

  std::ofstream out(outputPath, std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open output path for writing");
  }

  out << json << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  BenchmarkParameters parameters;
  std::string algorithmName;
  std::string datasetModeName = "memory";
  std::string excludeSelfValue = "true";
  std::string queryDatasetPath;
  std::string outputPath;

  CLI::App app{"Benchmark one algorithm/configuration and emit JSON."};
  app.set_help_flag("-h,--help", "Show this help message and exit.");

  app.add_option("--algorithm", algorithmName,
                 "Benchmark algorithm: bruteforce or vamana.")
      ->required()
      ->transform(CLI::CheckedTransformer(
          {{"bruteforce", "bruteforce"},
           {"brute-force", "bruteforce"},
           {"vamana", "vamana"}},
          CLI::ignore_case));
  app.add_option("--dataset", parameters.datasetPath, "Base dataset path.")
      ->required();
  app.add_option("--query-dataset", queryDatasetPath, "Optional query dataset.");
  app.add_option("--dataset-mode", datasetModeName,
                 "Dataset loading mode: file or memory.")
      ->capture_default_str()
      ->transform(CLI::CheckedTransformer({{"file", "file"},
                                           {"memory", "memory"}},
                                          CLI::ignore_case));
  app.add_option("--query-count", parameters.queryCount,
                 "Number of sampled queries. Default: min(100, N).");
  app.add_option("--k", parameters.k, "Recall/search depth.")
      ->capture_default_str();
  app.add_option("--seed", parameters.seed, "Deterministic workload seed.")
      ->capture_default_str();
  app.add_option("--exclude-self", excludeSelfValue,
                 "Exclude query id when queries come from the base dataset.")
      ->capture_default_str()
      ->transform(CLI::CheckedTransformer({{"true", "true"},
                                           {"false", "false"}},
                                          CLI::ignore_case));
  app.add_option("--artifact-dir", parameters.artifactDir,
                 "Directory for saved benchmark artifacts.")
      ->capture_default_str();
  app.add_option("--degree-threshold", parameters.degreeThreshold,
                 "Vamana R parameter.")
      ->capture_default_str();
  app.add_option("--search-list-size", parameters.searchListSize,
                 "Vamana L parameter.")
      ->capture_default_str();
  app.add_option("--distance-threshold", parameters.distanceThreshold,
                 "Vamana alpha parameter.")
      ->capture_default_str();
  app.add_option("--output", outputPath,
                 "Write JSON output to a file instead of stdout.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &error) {
    return app.exit(error);
  }

  try {
    parameters.algorithm = benchmarkAlgorithmFromName(algorithmName);
    parameters.datasetMode = benchmarkDataSetModeFromName(datasetModeName);
    parameters.excludeSelf = parseBoolValue(excludeSelfValue);
    if (!queryDatasetPath.empty()) {
      parameters.queryDatasetPath = queryDatasetPath;
    }

    const BenchmarkResult result = runBenchmark(parameters);
    const std::string json = benchmarkResultToJson(result);
    if (!outputPath.empty()) {
      writeBenchmarkResult(json, outputPath);
      return 0;
    }

    std::cout << json << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "sembed_benchmark: " << error.what() << '\n';
    return 1;
  }
}
