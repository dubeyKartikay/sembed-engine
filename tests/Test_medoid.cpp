#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "dataset.hpp"
#include "medoid.hpp"
#include "test_utils.hpp"
#include "vamana.hpp"

namespace {

std::unique_ptr<FlatDataSet> makeFlatDataset(
    uint64_t dims, const std::vector<std::vector<float>>& vectors) {
  auto ds = std::make_unique<FlatDataSet>(dims, vectors.size());
  for (uint64_t i = 0; i < vectors.size(); ++i) {
    ds->setVectorByIndex(i, static_cast<int64_t>(i), vectors[i].data(), dims);
  }
  return ds;
}

TEST(MedoidTest, EmptyDatasetReturnsNullopt) {
  FlatDataSet ds(4);
  EXPECT_EQ(sembed_internal::computeMedoidCpu(ds), std::nullopt);
  EXPECT_EQ(sembed_internal::computeMedoid(ds), std::nullopt);
}

TEST(MedoidTest, SinglePointReturnsZero) {
  auto ds = makeFlatDataset(3, {{1.0F, 2.0F, 3.0F}});
  EXPECT_EQ(sembed_internal::computeMedoidCpu(*ds), OptionalNodeId{0});
  EXPECT_EQ(sembed_internal::computeMedoid(*ds), OptionalNodeId{0});
}

TEST(MedoidTest, SelectsPointClosestToCentroidRatherThanMidpointIndex) {
  // Centroid is (0, 0, 0). Node 1 is at the origin, while midpoint index 2 is
  // a far outlier.
  auto ds = makeFlatDataset(3, {
      {-10.0F, 0.0F, 0.0F},
      {0.0F, 0.0F, 0.0F},
      {100.0F, 100.0F, 100.0F},
      {-90.0F, -100.0F, -100.0F},
      {0.0F, 0.0F, 0.0F},
  });
  // Exact tie between nodes 1 and 4 -> deterministic smallest index 1.
  EXPECT_EQ(sembed_internal::computeMedoidCpu(*ds), OptionalNodeId{1});
  EXPECT_EQ(sembed_internal::computeMedoid(*ds), OptionalNodeId{1});

  Vamana vamana(std::move(ds), /*degreeThreshold=*/2,
                /*distanceThreshold=*/1.2F, /*searchListSize=*/4);
  EXPECT_EQ(vamana.getMedoid(), OptionalNodeId{1});
}

TEST(MedoidTest, MatchesCpuOnGloveFixture) {
  FlatDataSet ds(testutils::embeddingFixturePath("gvec.bin"));
  const OptionalNodeId cpuMedoid = sembed_internal::computeMedoidCpu(ds);
  ASSERT_TRUE(cpuMedoid.has_value());
  EXPECT_EQ(sembed_internal::computeMedoid(ds), cpuMedoid);

  auto dsForVamana =
      std::make_unique<FlatDataSet>(testutils::embeddingFixturePath("gvec.bin"));
  Vamana vamana(std::move(dsForVamana), /*degreeThreshold=*/16,
                /*distanceThreshold=*/1.2F, /*searchListSize=*/32);
  EXPECT_EQ(vamana.getMedoid(), cpuMedoid);
}

#ifdef SEMBED_ENABLE_METAL
TEST(MedoidTest, MetalMatchesCpuAcrossShapesAndThreadgroups) {
  for (const uint64_t dims : {1ULL, 3ULL, 50ULL, 137ULL}) {
    const uint64_t n = 4097;
    std::vector<std::vector<float>> vectors(n, std::vector<float>(dims));
    for (uint64_t i = 0; i < n; ++i) {
      for (uint64_t d = 0; d < dims; ++d) {
        vectors[i][d] =
            std::sin(static_cast<float>(i * 17 + d * 31) * 0.01F) * 5.0F;
      }
    }
    // Place a clear geometric center at an arbitrary non-midpoint index.
    const uint64_t targetIdx = 1234;
    for (uint64_t d = 0; d < dims; ++d) {
      vectors[targetIdx][d] = 0.0F;
    }

    auto ds = makeFlatDataset(dims, vectors);
    const OptionalNodeId cpu = sembed_internal::computeMedoidCpu(*ds);
    const OptionalNodeId gpu = sembed_internal::computeMedoidMetal(*ds);
    ASSERT_TRUE(cpu.has_value());
    EXPECT_EQ(gpu, cpu) << "mismatch for dims=" << dims;
  }
}

TEST(MedoidTest, MetalMatchesCpuOnGloveFixture) {
  FlatDataSet ds(testutils::embeddingFixturePath("gvec.bin"));
  EXPECT_EQ(sembed_internal::computeMedoidMetal(ds),
            sembed_internal::computeMedoidCpu(ds));
}
#endif

}  // namespace
