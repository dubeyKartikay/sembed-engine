#include "cli_workflow.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <sembed/sembed.hpp>

namespace {

using Json = nlohmann::json;

std::vector<float> parseVector(const Json& value) {
  if (!value.is_array()) {
    throw std::invalid_argument("query vector must be a JSON array");
  }
  std::vector<float> vector;
  vector.reserve(value.size());
  for (const Json& component : value) {
    if (!component.is_number()) {
      throw std::invalid_argument("query vector components must be numbers");
    }
    vector.push_back(component.get<float>());
  }
  return vector;
}

Json resultsJson(const sembed::Index& index, const std::vector<float>& vector,
                 const sembed::QueryConfig& config) {
  const std::vector<sembed::Neighbor> neighbors = sembed::query(
      index, sembed::FloatVectorView(vector), config);
  Json results = Json::array();
  for (const sembed::Neighbor& neighbor : neighbors) {
    results.push_back({{"node", neighbor.node},
                       {"record_id", neighbor.recordId},
                       {"distance", neighbor.distance}});
  }
  return results;
}

Json queryResponse(const sembed::Index& index, const Json& request,
                   const sembed::QueryConfig& defaults) {
  const Json& vectorJson =
      request.is_object() ? request.at("vector") : request;
  const std::vector<float> vector = parseVector(vectorJson);

  sembed::QueryConfig config = defaults;
  if (request.is_object()) {
    config.k = request.value("k", config.k);
    config.searchListSize =
        request.value("search_list_size", config.searchListSize);
  }

  Json response = {{"results", resultsJson(index, vector, config)}};
  if (request.is_object() && request.contains("id")) {
    response["id"] = request["id"];
  }
  return response;
}

void runStreamingQueries(const sembed::Index& index,
                         const sembed::QueryConfig& defaults) {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    try {
      const Json request = Json::parse(line);
      std::cout << queryResponse(index, request, defaults).dump() << '\n'
                << std::flush;
    } catch (const std::exception& error) {
      std::cout << Json({{"error", error.what()}}).dump() << '\n'
                << std::flush;
    }
  }
}

}  // namespace

int runSembedCli(int argc, char** argv) {
  std::filesystem::path datasetPath;
  std::filesystem::path outputPath;
  std::filesystem::path indexPath;
  std::string vectorJson;
  bool stdinJsonl = false;
  sembed::IndexConfig indexConfig;
  sembed::QueryConfig queryConfig;

  CLI::App app{"Build and query self-contained Vamana indexes."};
  app.set_help_flag("-h,--help", "Show this help message and exit.");

  CLI::App* indexCommand =
      app.add_subcommand("index", "Build a self-contained index.");
  indexCommand->add_option("--dataset", datasetPath, "Binary dataset path.")
      ->required();
  indexCommand->add_option("--output", outputPath, "Output .sembed path.")
      ->required();
  indexCommand
      ->add_option("--degree-threshold", indexConfig.degreeThreshold,
                   "Maximum graph out-degree.")
      ->capture_default_str();
  indexCommand
      ->add_option("--search-list-size", indexConfig.searchListSize,
                   "Candidate list size used while building.")
      ->capture_default_str();
  indexCommand
      ->add_option("--distance-threshold", indexConfig.distanceThreshold,
                   "Vamana alpha pruning parameter.")
      ->capture_default_str();

  CLI::App* queryCommand =
      app.add_subcommand("query", "Query a self-contained index.");
  queryCommand->add_option("--index", indexPath, "Input .sembed path.")
      ->required();
  queryCommand->add_option(
      "--vector", vectorJson,
      "Full query vector encoded as a JSON array, for example '[1,2,3]'.");
  queryCommand->add_flag(
      "--stdin-jsonl", stdinJsonl,
      "Read JSON vectors or {id,vector} objects from stdin, one per line.");
  queryCommand->add_option("--k", queryConfig.k, "Number of neighbors.")
      ->capture_default_str();
  queryCommand
      ->add_option("--search-list-size", queryConfig.searchListSize,
                   "Candidate list size; 0 reuses the index setting.")
      ->capture_default_str();

  app.require_subcommand(1);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  }

  try {
    if (*indexCommand) {
      sembed::Index index = sembed::buildIndex(datasetPath, indexConfig);
      index.save(outputPath);
      std::cout << Json({{"command", "index"},
                         {"index", outputPath.string()},
                         {"records", index.size()},
                         {"dimensions", index.dimensions()},
                         {"config",
                          {{"degree_threshold", indexConfig.degreeThreshold},
                           {"search_list_size", indexConfig.searchListSize},
                           {"distance_threshold",
                            indexConfig.distanceThreshold}}}})
                       .dump(2)
                << '\n';
      return 0;
    }

    if (*queryCommand) {
      const bool hasVector = !vectorJson.empty();
      if (stdinJsonl == hasVector) {
        throw std::invalid_argument(
            "provide exactly one of --vector or --stdin-jsonl");
      }
      const sembed::Index index = sembed::Index::load(indexPath);
      if (stdinJsonl) {
        runStreamingQueries(index, queryConfig);
        return 0;
      }

      const Json request = Json::parse(vectorJson);
      Json response = queryResponse(index, request, queryConfig);
      response["command"] = "query";
      response["index"] = indexPath.string();
      std::cout << response.dump(2) << '\n';
      return 0;
    }
  } catch (const std::exception& error) {
    std::cerr << "sembed: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
