// Focused regressions around random utility behavior and exact-recall search
// expectations.

#include "HDVector.hpp"
#include "dataset.hpp"
#include "utils.hpp"
#include "vamana.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace {

std::string sanitize(std::string value) {
  for (char &ch : value) {
    const bool is_alnum = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                          (ch >= '0' && ch <= '9');
    if (!is_alnum && ch != '_' && ch != '-') {
      ch = '_';
    }
  }
  return value;
}

std::filesystem::path fixtureDir() {
  const auto dir = std::filesystem::current_path() / "build" / "test-fixtures";
  std::filesystem::create_directories(dir);
  return dir;
}

std::filesystem::path uniqueFixturePath(const std::string &tag) {
  const auto *test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string suite =
      test_info ? sanitize(test_info->test_suite_name()) : "unknown_suite";
  const std::string name =
      test_info ? sanitize(test_info->name()) : "unknown_test";
  return fixtureDir() /
         ("random_search_" + suite + "_" + name + "_" + tag + ".bin");
}

std::filesystem::path writeDatasetFile(
    const std::filesystem::path &path, long long n, long long storedDimensions,
    const std::vector<std::vector<float>> &rows) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open dataset fixture for writing");
  }
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(&storedDimensions),
            sizeof(storedDimensions));
  for (const auto &row : rows) {
    out.write(reinterpret_cast<const char *>(row.data()),
              static_cast<std::streamsize>(row.size() * sizeof(float)));
  }
  return path;
}

struct ScopedFile {
  std::filesystem::path path;

  ~ScopedFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

} // namespace

TEST(RandomUtilsRegression, GetRandomNumberVariesAcrossCalls) {
  std::set<int> observed;
  for (int trial = 0; trial < 50; ++trial) {
    observed.insert(getRandomNumber(0, 1000));
  }

  EXPECT_GT(observed.size(), 1U)
      << "getRandomNumber reseeds on every call, so callers always get the "
         "same value";
}

TEST(RandomUtilsRegression, GenerateRandomNumbersReturnsRequestedUniqueCount) {
  std::srand(0);

  constexpr int n = 8;
  constexpr int k = 7;
  const auto result = generateRandomNumbers(k, n, /*blackList=*/0);

  ASSERT_EQ(static_cast<int>(result.size()), k)
      << "collision handling dropped values instead of retrying until k "
         "unique neighbours were produced";

  const std::unordered_set<int> unique(result.begin(), result.end());
  EXPECT_EQ(unique.size(), result.size());
  EXPECT_EQ(unique.count(0), 0U);
}

TEST(VamanaSearchRegression, ExactRecallWhenSearchListEqualsDatasetSize) {
  const auto path = uniqueFixturePath("exact_recall");
  ScopedFile cleanup{path};

  std::vector<std::vector<float>> rows;
  for (int i = 0; i < 15; ++i) {
    rows.push_back({static_cast<float>(i),
                    static_cast<float>(i) * 2.0f + 0.5f,
                    static_cast<float>(i) * -0.3f});
  }
  writeDatasetFile(path, rows.size(), 3, rows);

  std::srand(0);
  auto ds = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(ds), 5);
  v.setSeachListSize(15);

  HDVector q(std::vector<float>{7.0f, 14.5f});
  const SearchResults r = v.greedySearch(q, 1);
  ASSERT_EQ(r.approximateNN.size(), 1U);

  const auto rec = v.m_dataSet->getRecordViewByIndex(r.approximateNN[0]);
  EXPECT_FLOAT_EQ(HDVector::distance(q, *rec.vector), 0.0f)
      << "with L == N the exact nearest neighbour should be recoverable";
}
