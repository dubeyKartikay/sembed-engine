#include "dataset.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

namespace {

std::string sanitizePathComponent(std::string value) {
  for (char &ch : value) {
    const bool is_alnum = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                          (ch >= '0' && ch <= '9');
    if (!is_alnum && ch != '_' && ch != '-') {
      ch = '_';
    }
  }
  return value;
}

struct TestFixtureData {
  long long n = 4;
  long long dimensions = 3;
  std::vector<std::vector<float>> rows = {
      {100.0f, 1.0f, 2.0f},
      {110.0f, 11.0f, 12.0f},
      {120.0f, 21.0f, 22.0f},
      {130.0f, 31.0f, 32.0f},
  };
};

std::filesystem::path makeDatasetFile(const std::string &name,
                                      const TestFixtureData &fixture) {
  const auto fixture_dir =
      std::filesystem::current_path() / "build" / "test-fixtures";
  std::filesystem::create_directories(fixture_dir);
  const auto path = fixture_dir / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("failed to create temporary dataset fixture");
  }

  out.write(reinterpret_cast<const char *>(&fixture.n), sizeof(fixture.n));
  out.write(reinterpret_cast<const char *>(&fixture.dimensions),
            sizeof(fixture.dimensions));
  for (const auto &row : fixture.rows) {
    out.write(reinterpret_cast<const char *>(row.data()),
              static_cast<std::streamsize>(row.size() * sizeof(float)));
  }

  return path;
}

template <typename DataSetType> class DataSetApiTest : public ::testing::Test {
protected:
  TestFixtureData fixture;
  std::filesystem::path datasetPath;

  void SetUp() override {
    const auto *suite_name = ::testing::UnitTest::GetInstance()
                                 ->current_test_info()
                                 ->test_suite_name();
    const auto *test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    datasetPath = makeDatasetFile(
        std::string("sembed_") + sanitizePathComponent(suite_name) + "_" +
            sanitizePathComponent(test_name) + ".bin",
        fixture);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove(datasetPath, ec);
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
  EXPECT_EQ(dataSet->getDimentions(), this->fixture.dimensions - 1);
}

TYPED_TEST(DataSetApiTest, ReturnsRecordViewsByIndex) {
  auto dataSet = this->makeDataSet();

  for (int row_index = 0; row_index < static_cast<int>(this->fixture.rows.size());
       ++row_index) {
    auto record = dataSet->getRecordViewByIndex(row_index);
    ASSERT_NE(record.vector, nullptr);
    EXPECT_EQ(record.recordId,
              static_cast<long long>(this->fixture.rows[row_index][0]));
    EXPECT_EQ(record.vector->getDimention(), this->fixture.dimensions - 1);

    for (int dim = 0; dim < record.vector->getDimention(); ++dim) {
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

  for (int offset = 0; offset < static_cast<int>(records->size()); ++offset) {
    const RecordView &record = records->at(offset);
    ASSERT_NE(record.vector, nullptr);
    EXPECT_EQ(record.recordId,
              static_cast<long long>(this->fixture.rows[offset + 1][0]));
    ASSERT_EQ(record.vector->getDimention(), this->fixture.dimensions - 1);
    for (int dim = 0; dim < record.vector->getDimention(); ++dim) {
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

  for (int offset = 0; offset < static_cast<int>(vectors->size()); ++offset) {
    ASSERT_NE(vectors->at(offset), nullptr);
    for (int dim = 0; dim < vectors->at(offset)->getDimention(); ++dim) {
      EXPECT_FLOAT_EQ((*vectors->at(offset))[dim],
                      this->fixture.rows[offset + 1][dim + 1]);
    }
  }
}

TYPED_TEST(DataSetApiTest, RejectsOutOfBoundsRecordIndex) {
  auto dataSet = this->makeDataSet();

  EXPECT_THROW((void)dataSet->getRecordViewByIndex(-1), std::out_of_range);
  EXPECT_THROW((void)dataSet->getRecordViewByIndex(this->fixture.rows.size()),
               std::out_of_range);
}

TYPED_TEST(DataSetApiTest, RejectsOutOfBoundsRecordRanges) {
  auto dataSet = this->makeDataSet();

  EXPECT_THROW((void)dataSet->getNRecordViewsFromIndex(-1, 1),
               std::out_of_range);
  EXPECT_THROW((void)dataSet->getNRecordViewsFromIndex(3, 2),
               std::out_of_range);
  EXPECT_THROW((void)dataSet->getNRecordViewsFromIndex(1, -1),
               std::out_of_range);
}

} // namespace
