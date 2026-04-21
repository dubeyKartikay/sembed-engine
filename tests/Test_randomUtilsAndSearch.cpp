// Focused regressions around random utility behavior and exact-recall search
// expectations.

#include "HDVector.hpp"
#include "dataset.hpp"
#include "test_utils.hpp"
#include "utils.hpp"
#include "vamana.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

namespace {
} // namespace

TEST(RandomUtilsRegression, GetRandomNumberVariesAcrossCalls) {
  std::set<int64_t> observed;
  for (int64_t trial = 0; trial < 50; ++trial) {
    observed.insert(getRandomNumber(0, 1000));
  }

  EXPECT_GT(observed.size(), 1U)
      << "getRandomNumber reseeds on every call, so callers always get the "
         "same value";
}

TEST(RandomUtilsRegression, GenerateRandomNumbersReturnsRequestedUniqueCount) {
  std::srand(0);

  constexpr uint64_t n = 8;
  constexpr uint64_t k = 7;
  const auto result = generateRandomNumbers(k, n, /*blackList=*/0);

  ASSERT_EQ(static_cast<uint64_t>(result.size()), k)
      << "collision handling dropped values instead of retrying until k "
         "unique neighbours were produced";

  const std::unordered_set<NodeId> unique(result.begin(), result.end());
  EXPECT_EQ(unique.size(), result.size());
  EXPECT_EQ(unique.count(0), 0U);
}

TEST(VamanaSearchRegression, ExactRecallWhenSearchListEqualsDatasetSize) {
  const auto path = testutils::uniqueFixturePath("random_search", "exact_recall");
  testutils::ScopedPathCleanup cleanup{path};

  std::vector<std::vector<float>> rows;
  for (int64_t i = 0; i < 15; ++i) {
    rows.push_back({static_cast<float>(i),
                    static_cast<float>(i) * 2.0f + 0.5f,
                    static_cast<float>(i) * -0.3f});
  }
  testutils::writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 5);
  v.setSeachListSize(15);

  HDVector q(std::vector<float>{14.5f, -2.1f});
  const SearchResults r = v.greedySearch(q, 1);
  ASSERT_EQ(r.approximateNN.size(), 1U);

  const auto rec = v.m_dataSet->getRecordViewByIndex(r.approximateNN[0]);
  EXPECT_NEAR(HDVector::distance(q, *rec.vector), 0.0f, 1.0e-6f)
      << "with L == N the exact nearest neighbour should be recoverable";
}
