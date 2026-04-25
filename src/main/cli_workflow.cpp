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
#include "vector_view.hpp"

namespace {

using Json = nlohmann::json;

std::unique_ptr<DataSet> makeDataSet(const std::filesystem::path &path) {
  return std::make_unique<FlatDataSet>(path);
}

void ensureParentDirectoryExists(const std::filesystem::path &path) {
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

Json baseDataSetJson(DataSet &dataSet, const std::filesystem::path &path) {
  return {{"path", path.string()},
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
          {"medoid",
           graph.getMedoid() ? Json(*graph.getMedoid()) : Json(nullptr)},
          {"total_edges", totalEdges},
          {"average_out_degree", averageOutDegree},
          {"max_out_degree", maxOutDegree}};
}

Json recordResultJson(const RecordView &queryRecord,
                      const RecordView &resultRecord, NodeId node) {
  return {{"node", node},
          {"record_id", resultRecord.recordId},
          {"distance", euclideanDistance(queryRecord.values,
                                          resultRecord.values)}};
}

}  // namespace

std::string buildIndexWorkflow(const BuildIndexOptions &options) {
  std::unique_ptr<DataSet> dataSet = makeDataSet(options.datasetPath);
  const Json dataSetJson = baseDataSetJson(*dataSet, options.datasetPath);

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
  std::unique_ptr<DataSet> dataSet = makeDataSet(options.datasetPath);
  if (options.queryNode >= dataSet->getN()) {
    throw std::out_of_range("query node is outside dataset bounds");
  }

  Vamana index(std::move(dataSet), options.indexPath);
  index.setSearchListSize(static_cast<int64_t>(options.searchListSize));

  const RecordView queryRecord = index.getRecordViewByIndex(options.queryNode);
  const SearchResults searchResults =
      index.greedySearch(queryRecord.values, options.k);

  Json results = Json::array();
  for (NodeId node : searchResults.approximateNN) {
    const RecordView resultRecord = index.getRecordViewByIndex(node);
    results.push_back(recordResultJson(queryRecord, resultRecord, node));
  }

  Json output = {
      {"command", "query-index"},
      {"dataset", {{"path", options.datasetPath.string()}}},
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
    std::unique_ptr<DataSet> dataSet = makeDataSet(*options.datasetPath);
    if (dataSet->getN() != graph.getNodeCount()) {
      throw std::runtime_error(
          "dataset record count does not match graph node count");
    }
    output["dataset"] = baseDataSetJson(*dataSet, *options.datasetPath);
  }

  return output.dump(2);
}

int runSembedCli(int argc, char **argv) {
  BuildIndexOptions buildOptions;
  QueryIndexOptions queryOptions;
  InspectIndexOptions inspectOptions;

  CLI::App app{"Build, query, and inspect sembed indexes."};
  app.set_help_flag("-h,--help", "Show this help message and exit.");

  CLI::App *buildIndex =
      app.add_subcommand("build-index", "Build and save a Vamana index.");
  buildIndex->add_option("--dataset", buildOptions.datasetPath,
                         "Dataset path.")
      ->required();
  buildIndex->add_option("--degree-threshold", buildOptions.degreeThreshold,
                         "Maximum graph out-degree.")
      ->capture_default_str();
  buildIndex->add_option("--distance-threshold",
                         buildOptions.distanceThreshold,
                         "Vamana alpha parameter.")
      ->capture_default_str();
  buildIndex->add_option("--output", buildOptions.outputPath,
                         "Output graph path.")
      ->required();

  CLI::App *queryIndex =
      app.add_subcommand("query-index", "Query a saved Vamana index.");
  queryIndex->add_option("--dataset", queryOptions.datasetPath,
                         "Dataset path.")
      ->required();
  queryIndex->add_option("--index", queryOptions.indexPath,
                         "Saved graph path.")
      ->required();
  queryIndex->add_option("--query-node", queryOptions.queryNode,
                         "Dataset node to use as query.")
      ->required();
  queryIndex->add_option("--k", queryOptions.k, "Number of results.")
      ->capture_default_str();
  queryIndex->add_option("--search-list-size", queryOptions.searchListSize,
                         "Vamana search list size.")
      ->capture_default_str();

  CLI::App *inspectIndex =
      app.add_subcommand("inspect-index", "Inspect a saved Vamana index.");
  inspectIndex->add_option("--index", inspectOptions.indexPath,
                           "Saved graph path.")
      ->required();
  inspectIndex->add_option("--dataset", inspectOptions.datasetPath,
                           "Optional dataset path for consistency checks.");

  app.require_subcommand(1);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &error) {
    return app.exit(error);
  }

  try {
    if (*buildIndex) {
      std::cout << buildIndexWorkflow(buildOptions) << '\n';
      return 0;
    }
    if (*queryIndex) {
      std::cout << queryIndexWorkflow(queryOptions) << '\n';
      return 0;
    }
    if (*inspectIndex) {
      std::cout << inspectIndexWorkflow(inspectOptions) << '\n';
      return 0;
    }
  } catch (const std::exception &error) {
    std::cerr << "sembed: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
