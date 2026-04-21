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

template <typename DataSetType> class DataSetApiTest : public ::testing::Test {
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

  std::unique_ptr<DataSetType> makeDataSet() {
    return std::make_unique<DataSetType>(datasetPath);
  }
};

using DataSetImplementations = ::testing::Types<FileDataSet, InMemoryDataSet>;
TYPED_TEST_SUITE(DataSetApiTest, DataSetImplementations);

TYPED_TEST(DataSetApiTest, ReportsDatasetShape) {
  auto dataSet = this->makeDataSet();

  EXPECT_EQ(dataSet->getN(), this->fixture.n);
  EXPECT_EQ(dataSet->getDimensions(), this->fixture.dimensions - 1);
}

TYPED_TEST(DataSetApiTest, ReturnsRecordViewsByIndex) {
  auto dataSet = this->makeDataSet();

  for (int64_t row_index = 0;
       row_index < static_cast<int64_t>(this->fixture.rows.size());
       ++row_index) {
    auto record = dataSet->getRecordViewByIndex(row_index);
    ASSERT_NE(record.vector, nullptr);
    EXPECT_EQ(record.recordId,
              static_cast<int64_t>(this->fixture.rows[row_index][0]));
    EXPECT_EQ(record.vector->getDimension(), this->fixture.dimensions - 1);

    for (int64_t dim = 0;
         dim < static_cast<int64_t>(record.vector->getDimension()); ++dim) {
      EXPECT_FLOAT_EQ((*record.vector)[dim],
                      this->fixture.rows[row_index][dim + 1]);
    }
  }
}

TYPED_TEST(DataSetApiTest, ReturnsContiguousRecordRangesFromIndex) {
  auto dataSet = this->makeDataSet();

  auto records = dataSet->getNRecordViewsFromIndex(1, 2);

  ASSERT_NE(records, nullptr);
  ASSERT_EQ(records->size(), 2U);

  for (int64_t offset = 0;
       offset < static_cast<int64_t>(records->size()); ++offset) {
    const RecordView &record = records->at(offset);
    ASSERT_NE(record.vector, nullptr);
    EXPECT_EQ(record.recordId,
              static_cast<int64_t>(this->fixture.rows[offset + 1][0]));
    ASSERT_EQ(record.vector->getDimension(), this->fixture.dimensions - 1);
    for (int64_t dim = 0;
         dim < static_cast<int64_t>(record.vector->getDimension()); ++dim) {
      EXPECT_FLOAT_EQ((*record.vector)[dim],
                      this->fixture.rows[offset + 1][dim + 1]);
    }
  }
}

TYPED_TEST(DataSetApiTest, ReturnsContiguousVectorRangesFromIndex) {
  auto dataSet = this->makeDataSet();

  auto vectors = dataSet->getNHDVectorsFromIndex(1, 2);

  ASSERT_NE(vectors, nullptr);
  ASSERT_EQ(vectors->size(), 2U);

  for (int64_t offset = 0;
       offset < static_cast<int64_t>(vectors->size()); ++offset) {
    ASSERT_NE(vectors->at(offset), nullptr);
    for (int64_t dim = 0;
         dim < static_cast<int64_t>(vectors->at(offset)->getDimension());
         ++dim) {
      EXPECT_FLOAT_EQ((*vectors->at(offset))[dim],
                      this->fixture.rows[offset + 1][dim + 1]);
    }
  }
}

TYPED_TEST(DataSetApiTest, RejectsOutOfBoundsRecordIndex) {
  auto dataSet = this->makeDataSet();

  EXPECT_THROW((void)dataSet->getRecordViewByIndex(
                   std::numeric_limits<uint64_t>::max()),
               std::out_of_range);
  EXPECT_THROW((void)dataSet->getRecordViewByIndex(this->fixture.rows.size()),
               std::out_of_range);
}

TYPED_TEST(DataSetApiTest, RejectsOutOfBoundsRecordRanges) {
  auto dataSet = this->makeDataSet();

  EXPECT_THROW((void)dataSet->getNRecordViewsFromIndex(3, 2),
               std::out_of_range);
  EXPECT_THROW((void)dataSet->getNRecordViewsFromIndex(
                   std::numeric_limits<uint64_t>::max(), 1),
               std::out_of_range);
  EXPECT_THROW((void)dataSet->getNRecordViewsFromIndex(
                   1, std::numeric_limits<uint64_t>::max()),
               std::out_of_range);
}

} // namespace
