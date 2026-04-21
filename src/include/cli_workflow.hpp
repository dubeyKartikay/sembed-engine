#ifndef CLI_WORKFLOW
#define CLI_WORKFLOW

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

enum class CliDataSetMode {
  File,
  Memory,
};

struct BuildIndexOptions {
  std::filesystem::path datasetPath;
  std::filesystem::path outputPath;
  CliDataSetMode dataSetMode = CliDataSetMode::Memory;
  uint64_t degreeThreshold = 64;
  float distanceThreshold = 1.2f;
};

struct QueryIndexOptions {
  std::filesystem::path datasetPath;
  std::filesystem::path indexPath;
  CliDataSetMode dataSetMode = CliDataSetMode::Memory;
  uint64_t queryNode = 0;
  uint64_t k = 10;
  uint64_t searchListSize = 100;
};

struct InspectIndexOptions {
  std::filesystem::path indexPath;
  std::optional<std::filesystem::path> datasetPath;
  CliDataSetMode dataSetMode = CliDataSetMode::Memory;
};

std::string buildIndexWorkflow(const BuildIndexOptions &options);
std::string queryIndexWorkflow(const QueryIndexOptions &options);
std::string inspectIndexWorkflow(const InspectIndexOptions &options);
int runSembedCli(int argc, char **argv);

#endif  // CLI_WORKFLOW
