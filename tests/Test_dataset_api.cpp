#include "dataset.hpp"
#include "test_utils.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

namespace {

struct TestFixtureData {
  int64_t n = 4;
  int64_t dimensions = 3;
  std::vector<std::vector<float>> rows = {
      {100.0f, 1.0f, 2.0f},
      {110.0f, 11.0f, 12.0f},
      {120.0f, 21.0f, 22.0f},
      {130.0f, 31.0f, 32.0f},
  };
};

class DataSetApiTest : public ::testing::Test {
protected:
  TestFixtureData fixture;
  std::filesystem::path datasetPath;

  void SetUp() override {
    datasetPath = testutils::uniqueFixturePath("sembed", "dataset");
    testutils::writeDatasetFile(datasetPath, fixture.n, fixture.dimensions,
                                fixture.rows);
  }

  void TearDown() override {
    testutils::ScopedPathCleanup cleanup(datasetPath);
  }

  std::unique_ptr<FlatDataSet> makeDataSet() {
    return std::make_unique<FlatDataSet>(datasetPath);
  }
};

TEST_F(DataSetApiTest, ReportsDatasetShape) {
  auto dataSet = makeDataSet();

  EXPECT_EQ(dataSet->getN(), fixture.n);
  EXPECT_EQ(dataSet->getDimensions(), fixture.dimensions - 1);
}

TEST_F(DataSetApiTest, ReturnsRecordViewsByIndex) {
  auto dataSet = makeDataSet();

  for (int64_t rowIndex = 0;
       rowIndex < static_cast<int64_t>(fixture.rows.size()); ++rowIndex) {
    auto record = dataSet->getRecordViewByIndex(rowIndex);
    ASSERT_NE(record.values.data(), nullptr);
    EXPECT_EQ(record.recordId, static_cast<int64_t>(fixture.rows[rowIndex][0]));
    EXPECT_EQ(record.values.dimensions(), fixture.dimensions - 1);

    for (uint64_t dim = 0; dim < record.values.dimensions(); ++dim) {
      EXPECT_FLOAT_EQ(record.values[dim], fixture.rows[rowIndex][dim + 1]);
    }
  }
}

TEST_F(DataSetApiTest, ReturnsContiguousRecordRangesFromIndex) {
  auto dataSet = makeDataSet();

  auto records = dataSet->getRecordViewsFromIndex(1, 2);

  ASSERT_EQ(records.size(), 2U);
  for (uint64_t offset = 0; offset < records.size(); ++offset) {
    const RecordView &record = records.at(static_cast<size_t>(offset));
    ASSERT_NE(record.values.data(), nullptr);
    EXPECT_EQ(record.recordId,
              static_cast<int64_t>(fixture.rows[offset + 1][0]));
    ASSERT_EQ(record.values.dimensions(), fixture.dimensions - 1);
    for (uint64_t dim = 0; dim < record.values.dimensions(); ++dim) {
      EXPECT_FLOAT_EQ(record.values[dim], fixture.rows[offset + 1][dim + 1]);
    }
  }
}

TEST_F(DataSetApiTest, AllowsEmptyRangeAtEnd) {
  auto dataSet = makeDataSet();

  auto records = dataSet->getRecordViewsFromIndex(fixture.rows.size(), 0);
  EXPECT_TRUE(records.empty());
}

TEST_F(DataSetApiTest, RejectsOutOfBoundsRecordIndex) {
  auto dataSet = makeDataSet();

  EXPECT_THROW((void)dataSet->getRecordViewByIndex(
                   std::numeric_limits<uint64_t>::max()),
               std::out_of_range);
  EXPECT_THROW((void)dataSet->getRecordViewByIndex(fixture.rows.size()),
               std::out_of_range);
}

TEST_F(DataSetApiTest, RejectsOutOfBoundsRecordRanges) {
  auto dataSet = makeDataSet();

  EXPECT_THROW((void)dataSet->getRecordViewsFromIndex(3, 2),
               std::out_of_range);
  EXPECT_THROW((void)dataSet->getRecordViewsFromIndex(
                   std::numeric_limits<uint64_t>::max(), 1),
               std::out_of_range);
  EXPECT_THROW((void)dataSet->getRecordViewsFromIndex(
                   1, std::numeric_limits<uint64_t>::max()),
               std::out_of_range);
}

}  // namespace
