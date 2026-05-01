#include "dataset.hpp"
#include "test_utils.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {

template <typename DataSetType> class BinaryLoadingTest : public ::testing::Test {};

using DataSetImplementations = ::testing::Types<FlatDataSet>;
TYPED_TEST_SUITE(BinaryLoadingTest, DataSetImplementations);

TYPED_TEST(BinaryLoadingTest, LoadsGloveBinary) {
  TypeParam dataset(testutils::embeddingFixturePath("gvec.bin"));
  ASSERT_EQ(dataset.getN(), testutils::kGloveFixtureRows);
  ASSERT_EQ(dataset.getDimensions(), testutils::kGloveFixtureDimensions);
}

TYPED_TEST(BinaryLoadingTest, ReadsFirstVectorFromGloveBinary) {
  TypeParam dataset(testutils::embeddingFixturePath("gvec.bin"));
  RecordView record = dataset.getRecordViewByIndex(0);
  ASSERT_EQ(record.recordId, 0);
  ASSERT_NE(record.values.data(), nullptr);

  const auto &expected = testutils::firstGloveVector();
  for (uint64_t i = 0; i < dataset.getDimensions(); ++i) {
    EXPECT_EQ(record.values[i], expected.at(static_cast<size_t>(i)));
  }
}

TYPED_TEST(BinaryLoadingTest, ReadsLastVectorFromGloveBinary) {
  TypeParam dataset(testutils::embeddingFixturePath("gvec.bin"));
  RecordView record = dataset.getRecordViewByIndex(dataset.getN() - 1);
  ASSERT_EQ(record.recordId, dataset.getN() - 1);
  ASSERT_NE(record.values.data(), nullptr);

  const auto &expected = testutils::lastGloveVector();
  for (uint64_t i = 0; i < dataset.getDimensions(); ++i) {
    EXPECT_EQ(record.values[i], expected.at(static_cast<size_t>(i)));
  }
}

TYPED_TEST(BinaryLoadingTest, LoadsIdBlockThenColumnMajorFloatPayload) {
  const auto path =
      testutils::uniqueFixturePath("dataset_layout", "column_major");
  testutils::ScopedPathCleanup cleanup(path);

  const int64_t n = 2;
  const int64_t storedDimensions = 4;
  const int64_t ids[] = {42, 99};
  const float values[] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(&storedDimensions),
            sizeof(storedDimensions));
  out.write(reinterpret_cast<const char *>(ids), sizeof(ids));
  out.write(reinterpret_cast<const char *>(values), sizeof(values));
  out.close();

  TypeParam dataset(path);
  ASSERT_EQ(dataset.getN(), 2U);
  ASSERT_EQ(dataset.getDimensions(), 3U);

  RecordView first = dataset.getRecordViewByIndex(0);
  EXPECT_EQ(first.recordId, 42);
  EXPECT_FLOAT_EQ(first.values[0], 1.0f);
  EXPECT_FLOAT_EQ(first.values[1], 2.0f);
  EXPECT_FLOAT_EQ(first.values[2], 3.0f);

  RecordView second = dataset.getRecordViewByIndex(1);
  EXPECT_EQ(second.recordId, 99);
  EXPECT_FLOAT_EQ(second.values[0], 4.0f);
  EXPECT_FLOAT_EQ(second.values[1], 5.0f);
  EXPECT_FLOAT_EQ(second.values[2], 6.0f);
}

}  // namespace
