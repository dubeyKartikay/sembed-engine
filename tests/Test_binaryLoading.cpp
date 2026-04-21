#include "dataset.hpp"
#include "test_utils.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {

template <typename DataSetType>
class BinaryLoadingTest : public ::testing::Test {};

using DataSetImplementations = ::testing::Types<FileDataSet, InMemoryDataSet>;
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
  ASSERT_NE(record.vector, nullptr);

  const auto &expected = testutils::firstGloveVector();
  for (int64_t i = 0; i < static_cast<int64_t>(dataset.getDimensions()); ++i) {
    EXPECT_EQ((*record.vector)[i], expected.at(static_cast<size_t>(i)));
  }
}

TYPED_TEST(BinaryLoadingTest, ReadsLastVectorFromGloveBinary) {
  TypeParam dataset(testutils::embeddingFixturePath("gvec.bin"));
  RecordView record = dataset.getRecordViewByIndex(dataset.getN() - 1);
  ASSERT_EQ(record.recordId, dataset.getN() - 1);
  ASSERT_NE(record.vector, nullptr);

  const auto &expected = testutils::lastGloveVector();
  for (int64_t i = 0; i < static_cast<int64_t>(dataset.getDimensions()); ++i) {
    EXPECT_EQ((*record.vector)[i], expected.at(static_cast<size_t>(i)));
  }
}

}  // namespace
