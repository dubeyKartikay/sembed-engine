#include "cli_workflow.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "dataset.hpp"
#include "graph.hpp"
#include "vamana.hpp"

namespace {

using Json = nlohmann::json;

std::string dataSetModeName(CliDataSetMode mode) {
  switch (mode) {
    case CliDataSetMode::File:
      return "file";
    case CliDataSetMode::Memory:
      return "memory";
  }

  throw std::invalid_argument("unsupported dataset mode");
}

CliDataSetMode dataSetModeFromName(const std::string &value) {
  if (value == "file") {
    return CliDataSetMode::File;
  }
  if (value == "memory") {
    return CliDataSetMode::Memory;
  }

  throw std::invalid_argument("unsupported dataset mode: " + value);
}

std::unique_ptr<DataSet> makeDataSet(CliDataSetMode mode,
                                     const std::filesystem::path &path) {
  switch (mode) {
    case CliDataSetMode::File:
      return std::make_unique<FileDataSet>(path);
    case CliDataSetMode::Memory:
      return std::make_unique<InMemoryDataSet>(path);
  }

  throw std::invalid_argument("unsupported dataset mode");
}

void ensureParentDirectoryExists(const std::filesystem::path &path) {
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

Json baseDataSetJson(DataSet &dataSet, const std::filesystem::path &path,
                     CliDataSetMode mode) {
  return {{"path", path.string()},
          {"mode", dataSetModeName(mode)},
          {"records", dataSet.getN()},
          {"dimensions", dataSet.getDimensions()}};
}

Json graphSummaryJson(const Graph &graph) {
  uint64_t totalEdges = 0;
  uint64_t maxOutDegree = 0;
  for (NodeId node = 0; node < graph.getNodeCount(); ++node) {
    const uint64_t degree =
        static_cast<uint64_t>(graph.getOutNeighbors(node).size());
    totalEdges += degree;
    if (degree > maxOutDegree) {
      maxOutDegree = degree;
    }
  }

  const double averageOutDegree =
      graph.getNodeCount() == 0
          ? 0.0
          : static_cast<double>(totalEdges) /
                static_cast<double>(graph.getNodeCount());

  return {{"nodes", graph.getNodeCount()},
          {"degree_threshold", graph.getDegreeThreshold()},
          {"medoid", graph.getMedoid() ? Json(*graph.getMedoid()) : Json(nullptr)},
          {"total_edges", totalEdges},
          {"average_out_degree", averageOutDegree},
          {"max_out_degree", maxOutDegree}};
}

Json recordResultJson(const RecordView &queryRecord, const RecordView &resultRecord,
                      NodeId node) {
  return {{"node", node},
          {"record_id", resultRecord.recordId},
          {"distance",
           HDVector::distance(*queryRecord.vector, *resultRecord.vector)}};
}

}  // namespace

std::string buildIndexWorkflow(const BuildIndexOptions &options) {
  std::unique_ptr<DataSet> dataSet =
      makeDataSet(options.dataSetMode, options.datasetPath);
  const Json dataSetJson =
      baseDataSetJson(*dataSet, options.datasetPath, options.dataSetMode);

  Vamana index(std::move(dataSet), options.degreeThreshold,
               options.distanceThreshold);
  ensureParentDirectoryExists(options.outputPath);
  index.save(options.outputPath);

  Json output = {{"command", "build-index"},
                 {"dataset", dataSetJson},
                 {"index",
                  {{"path", options.outputPath.string()},
                   {"degree_threshold", index.getDegreeThreshold()},
                   {"distance_threshold", index.getDistanceThreshold()},
                   {"medoid",
                    index.getMedoid() ? Json(*index.getMedoid()) : Json(nullptr)}}}};
  return output.dump(2);
}

std::string queryIndexWorkflow(const QueryIndexOptions &options) {
  std::unique_ptr<DataSet> dataSet =
      makeDataSet(options.dataSetMode, options.datasetPath);
  if (options.queryNode >= dataSet->getN()) {
    throw std::out_of_range("query node is outside dataset bounds");
  }

  Vamana index(std::move(dataSet), options.indexPath);
  index.setSearchListSize(static_cast<int64_t>(options.searchListSize));

  const RecordView queryRecord = index.getRecordViewByIndex(options.queryNode);
  const SearchResults searchResults =
      index.greedySearch(*queryRecord.vector, options.k);

  Json results = Json::array();
  for (NodeId node : searchResults.approximateNN) {
    const RecordView resultRecord = index.getRecordViewByIndex(node);
    results.push_back(recordResultJson(queryRecord, resultRecord, node));
  }

  Json output = {
      {"command", "query-index"},
      {"dataset", {{"path", options.datasetPath.string()},
                   {"mode", dataSetModeName(options.dataSetMode)}}},
      {"index", {{"path", options.indexPath.string()}}},
      {"query",
       {{"node", options.queryNode},
        {"record_id", queryRecord.recordId},
        {"k", options.k},
        {"search_list_size", options.searchListSize}}},
      {"results", results},
      {"visited_nodes", searchResults.visited}};
  return output.dump(2);
}

std::string inspectIndexWorkflow(const InspectIndexOptions &options) {
  Graph graph(options.indexPath);

  Json output = {{"command", "inspect-index"},
                 {"index",
                  {{"path", options.indexPath.string()},
                   {"graph", graphSummaryJson(graph)}}}};

  if (options.datasetPath) {
    std::unique_ptr<DataSet> dataSet =
        makeDataSet(options.dataSetMode, *options.datasetPath);
    if (dataSet->getN() != graph.getNodeCount()) {
      throw std::runtime_error(
          "dataset record count does not match graph node count");
    }
    output["dataset"] =
        baseDataSetJson(*dataSet, *options.datasetPath, options.dataSetMode);
  }

  return output.dump(2);
}

int runSembedCli(int argc, char **argv) {
  BuildIndexOptions buildOptions;
  QueryIndexOptions queryOptions;
  InspectIndexOptions inspectOptions;

  std::string buildModeName = "memory";
  std::string queryModeName = "memory";
  std::string inspectModeName = "memory";

  CLI::App app{"Build, query, and inspect Vamana graph indexes."};
  app.set_help_flag("-h,--help", "Show this help message and exit.");

  auto *buildIndex = app.add_subcommand(
      "build-index", "Build a Vamana index and save the graph to disk.");
  buildIndex->add_option("--dataset", buildOptions.datasetPath,
                         "Dataset binary path.")
      ->required();
  buildIndex->add_option("--dataset-mode", buildModeName,
                         "Dataset loading mode: file or memory.")
      ->capture_default_str()
      ->transform(CLI::CheckedTransformer({{"file", "file"},
                                           {"memory", "memory"}},
                                          CLI::ignore_case));
  buildIndex->add_option("--degree-threshold", buildOptions.degreeThreshold,
                         "Vamana R parameter.")
      ->capture_default_str();
  buildIndex->add_option("--distance-threshold", buildOptions.distanceThreshold,
                         "Vamana alpha parameter.")
      ->capture_default_str();
  buildIndex->add_option("--output", buildOptions.outputPath,
                         "Graph output path.")
      ->required();

  auto *queryIndex = app.add_subcommand(
      "query-index", "Load a saved graph and query it using a dataset node.");
  queryIndex->add_option("--dataset", queryOptions.datasetPath,
                         "Dataset binary path.")
      ->required();
  queryIndex->add_option("--dataset-mode", queryModeName,
                         "Dataset loading mode: file or memory.")
      ->capture_default_str()
      ->transform(CLI::CheckedTransformer({{"file", "file"},
                                           {"memory", "memory"}},
                                          CLI::ignore_case));
  queryIndex->add_option("--index", queryOptions.indexPath, "Saved graph path.")
      ->required();
  queryIndex->add_option("--query-node", queryOptions.queryNode,
                         "Dataset node id to use as the query.")
      ->required();
  queryIndex->add_option("--k", queryOptions.k, "Number of neighbors to return.")
      ->capture_default_str();
  queryIndex->add_option("--search-list-size", queryOptions.searchListSize,
                         "Greedy-search candidate list size.")
      ->capture_default_str();

  auto *inspectIndex = app.add_subcommand(
      "inspect-index", "Report graph metadata for a saved index.");
  inspectIndex->add_option("--index", inspectOptions.indexPath, "Saved graph path.")
      ->required();
  inspectIndex->add_option("--dataset", inspectOptions.datasetPath,
                           "Optional dataset binary path for shape validation.");
  inspectIndex->add_option("--dataset-mode", inspectModeName,
                           "Dataset loading mode when --dataset is provided.")
      ->capture_default_str()
      ->transform(CLI::CheckedTransformer({{"file", "file"},
                                           {"memory", "memory"}},
                                          CLI::ignore_case));

  app.require_subcommand(1);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &error) {
    return app.exit(error);
  }

  try {
    if (*buildIndex) {
      buildOptions.dataSetMode = dataSetModeFromName(buildModeName);
      std::cout << buildIndexWorkflow(buildOptions) << '\n';
      return 0;
    }
    if (*queryIndex) {
      queryOptions.dataSetMode = dataSetModeFromName(queryModeName);
      std::cout << queryIndexWorkflow(queryOptions) << '\n';
      return 0;
    }

    inspectOptions.dataSetMode = dataSetModeFromName(inspectModeName);
    std::cout << inspectIndexWorkflow(inspectOptions) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "sembed: " << error.what() << '\n';
    return 1;
  }
}
