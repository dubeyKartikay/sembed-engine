#include "cli_workflow.hpp"

#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "test_utils.hpp"

namespace {

using Json = nlohmann::json;
using ScopedFile = testutils::ScopedPathCleanup;
using testutils::writeDatasetFile;

std::filesystem::path uniqueFixturePath(const std::string& tag,
                                        const std::string& extension = ".bin") {
  return testutils::uniqueFixturePath("cli", tag, extension);
}

int runCli(std::vector<std::string> arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (std::string& argument : arguments) {
    argv.push_back(argument.data());
  }
  return runSembedCli(static_cast<int>(argv.size()), argv.data());
}

TEST(CliWorkflow, IndexWritesSelfContainedIndexAndReportsMetadata) {
  const auto datasetPath = uniqueFixturePath("dataset");
  const auto indexPath = uniqueFixturePath("index", ".sembed");
  ScopedFile datasetCleanup(datasetPath);
  ScopedFile indexCleanup(indexPath);

  writeDatasetFile(datasetPath, 4, 3,
                   {{101.0f, 0.0f, 0.0f},
                    {102.0f, 1.0f, 0.0f},
                    {103.0f, 2.0f, 0.0f},
                    {104.0f, 3.0f, 0.0f}});

  testing::internal::CaptureStdout();
  const int result = runCli(
      {"sembed", "index", "--dataset", datasetPath.string(), "--output",
       indexPath.string(), "--degree-threshold", "2", "--search-list-size",
       "4"});
  const Json output = Json::parse(testing::internal::GetCapturedStdout());

  ASSERT_EQ(result, 0);
  ASSERT_TRUE(std::filesystem::exists(indexPath));
  EXPECT_EQ(output.at("command"), "index");
  EXPECT_EQ(output.at("records"), 4);
  EXPECT_EQ(output.at("dimensions"), 2);
  EXPECT_EQ(output.at("config").at("degree_threshold"), 2);
}

TEST(CliWorkflow, QueryAcceptsAFullVectorWithoutTheOriginalDataset) {
  const auto datasetPath = uniqueFixturePath("query_dataset");
  const auto indexPath = uniqueFixturePath("query_index", ".sembed");
  ScopedFile datasetCleanup(datasetPath);
  ScopedFile indexCleanup(indexPath);

  writeDatasetFile(datasetPath, 5, 3,
                   {{201.0f, 0.0f, 0.0f},
                    {202.0f, 1.0f, 0.0f},
                    {203.0f, 2.0f, 0.0f},
                    {204.0f, 8.0f, 8.0f},
                    {205.0f, 9.0f, 8.0f}});

  testing::internal::CaptureStdout();
  ASSERT_EQ(runCli({"sembed", "index", "--dataset", datasetPath.string(),
                    "--output", indexPath.string(), "--degree-threshold", "3",
                    "--search-list-size", "5"}),
            0);
  (void)testing::internal::GetCapturedStdout();
  ASSERT_TRUE(std::filesystem::remove(datasetPath));

  testing::internal::CaptureStdout();
  const int result =
      runCli({"sembed", "query", "--index", indexPath.string(), "--vector",
              "[2.1,0.0]", "--k", "3", "--search-list-size", "5"});
  const Json output = Json::parse(testing::internal::GetCapturedStdout());

  ASSERT_EQ(result, 0);
  ASSERT_EQ(output.at("command"), "query");
  ASSERT_EQ(output.at("results").size(), 3U);
  EXPECT_EQ(output.at("results").at(0).at("record_id"), 203);
}

TEST(CliWorkflow, QueryRejectsMismatchedVectorDimensions) {
  const auto datasetPath = uniqueFixturePath("dimensions_dataset");
  const auto indexPath = uniqueFixturePath("dimensions_index", ".sembed");
  ScopedFile datasetCleanup(datasetPath);
  ScopedFile indexCleanup(indexPath);

  writeDatasetFile(datasetPath, 2, 3,
                   {{301.0f, 0.0f, 0.0f}, {302.0f, 1.0f, 1.0f}});

  testing::internal::CaptureStdout();
  ASSERT_EQ(runCli({"sembed", "index", "--dataset", datasetPath.string(),
                    "--output", indexPath.string(), "--degree-threshold", "1",
                    "--search-list-size", "2"}),
            0);
  (void)testing::internal::GetCapturedStdout();

  testing::internal::CaptureStderr();
  const int result = runCli({"sembed", "query", "--index", indexPath.string(),
                             "--vector", "[1.0]"});
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, 1);
  EXPECT_NE(error.find("query dimensions must match the index"),
            std::string::npos);
}

}  // namespace
