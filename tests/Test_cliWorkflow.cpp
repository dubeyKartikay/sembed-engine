#include "cli_workflow.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "test_utils.hpp"

namespace {

using Json = nlohmann::json;
using ScopedFile = testutils::ScopedPathCleanup;
using testutils::writeDatasetFile;

std::filesystem::path uniqueFixturePath(const std::string &tag,
                                        const std::string &extension = ".bin") {
  return testutils::uniqueFixturePath("cli", tag, extension);
}

TEST(CliWorkflow, BuildIndexWritesGraphAndReportsMetadata) {
  const auto datasetPath = uniqueFixturePath("dataset");
  const auto indexPath = uniqueFixturePath("index", ".graph");
  ScopedFile datasetCleanup(datasetPath);
  ScopedFile indexCleanup(indexPath);

  writeDatasetFile(datasetPath, 4, 3,
                   {{0.0f, 0.0f, 0.0f},
                    {1.0f, 1.0f, 0.0f},
                    {2.0f, 2.0f, 0.0f},
                    {3.0f, 3.0f, 0.0f}});

  BuildIndexOptions options;
  options.datasetPath = datasetPath;
  options.outputPath = indexPath;
  options.degreeThreshold = 2;
  options.distanceThreshold = 1.2f;

  const Json result = Json::parse(buildIndexWorkflow(options));

  ASSERT_TRUE(std::filesystem::exists(indexPath));
  EXPECT_EQ(result.at("command"), "build-index");
  EXPECT_EQ(result.at("dataset").at("records"), 4);
  EXPECT_EQ(result.at("dataset").at("dimensions"), 2);
  EXPECT_EQ(result.at("index").at("degree_threshold"), 2);
}

TEST(CliWorkflow, QueryIndexReturnsNeighborResults) {
  const auto datasetPath = uniqueFixturePath("query_dataset");
  const auto indexPath = uniqueFixturePath("query_index", ".graph");
  ScopedFile datasetCleanup(datasetPath);
  ScopedFile indexCleanup(indexPath);

  writeDatasetFile(datasetPath, 5, 3,
                   {{0.0f, 0.0f, 0.0f},
                    {1.0f, 1.0f, 0.0f},
                    {2.0f, 2.0f, 0.0f},
                    {3.0f, 3.0f, 0.0f},
                    {4.0f, 4.0f, 0.0f}});

  BuildIndexOptions buildOptions;
  buildOptions.datasetPath = datasetPath;
  buildOptions.outputPath = indexPath;
  buildOptions.degreeThreshold = 3;
  buildOptions.distanceThreshold = 1.2f;
  (void)buildIndexWorkflow(buildOptions);

  QueryIndexOptions queryOptions;
  queryOptions.datasetPath = datasetPath;
  queryOptions.indexPath = indexPath;
  queryOptions.queryNode = 2;
  queryOptions.k = 3;
  queryOptions.searchListSize = 5;

  const Json result = Json::parse(queryIndexWorkflow(queryOptions));

  ASSERT_EQ(result.at("command"), "query-index");
  ASSERT_EQ(result.at("query").at("node"), 2);
  ASSERT_EQ(result.at("results").size(), 3U);
  EXPECT_EQ(result.at("results").at(0).at("node"), 2);
}

TEST(CliWorkflow, InspectIndexReportsGraphAndDatasetShape) {
  const auto datasetPath = uniqueFixturePath("inspect_dataset");
  const auto indexPath = uniqueFixturePath("inspect_index", ".graph");
  ScopedFile datasetCleanup(datasetPath);
  ScopedFile indexCleanup(indexPath);

  writeDatasetFile(datasetPath, 3, 4,
                   {{0.0f, 1.0f, 2.0f, 3.0f},
                    {1.0f, 4.0f, 5.0f, 6.0f},
                    {2.0f, 7.0f, 8.0f, 9.0f}});

  BuildIndexOptions buildOptions;
  buildOptions.datasetPath = datasetPath;
  buildOptions.outputPath = indexPath;
  buildOptions.degreeThreshold = 2;
  buildOptions.distanceThreshold = 1.2f;
  (void)buildIndexWorkflow(buildOptions);

  InspectIndexOptions inspectOptions;
  inspectOptions.indexPath = indexPath;
  inspectOptions.datasetPath = datasetPath;

  const Json result = Json::parse(inspectIndexWorkflow(inspectOptions));

  EXPECT_EQ(result.at("command"), "inspect-index");
  EXPECT_EQ(result.at("index").at("graph").at("nodes"), 3);
  EXPECT_EQ(result.at("dataset").at("dimensions"), 3);
}

}  // namespace
