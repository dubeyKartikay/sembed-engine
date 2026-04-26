#ifndef CLI_WORKFLOW
#define CLI_WORKFLOW

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

struct BuildIndexOptions {
  std::filesystem::path datasetPath;
  std::filesystem::path outputPath;
  uint64_t degreeThreshold = 64;
  float distanceThreshold = 1.2f;
};

struct QueryIndexOptions {
  std::filesystem::path datasetPath;
  std::filesystem::path indexPath;
  uint64_t queryNode = 0;
  uint64_t k = 10;
  uint64_t searchListSize = 100;
};

struct InspectIndexOptions {
  std::filesystem::path indexPath;
  std::optional<std::filesystem::path> datasetPath;
};

std::string buildIndexWorkflow(const BuildIndexOptions &options);
std::string queryIndexWorkflow(const QueryIndexOptions &options);
std::string inspectIndexWorkflow(const InspectIndexOptions &options);
int runSembedCli(int argc, char **argv);

#endif  // CLI_WORKFLOW
