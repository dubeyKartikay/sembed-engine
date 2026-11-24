#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "dataset.hpp"
#include "vamana.hpp"

namespace {

std::unique_ptr<FlatDataSet> makeDataset(
    uint64_t dims, const std::vector<std::vector<float>>& vectors) {
  auto ds = std::make_unique<FlatDataSet>(dims, vectors.size());
  for (uint64_t i = 0; i < vectors.size(); ++i) {
    ds->setVectorByIndex(i, static_cast<int64_t>(i), vectors[i].data(), dims);
  }
  return ds;
}

TEST(VamanaPruneTest, ProgressiveCurAlphaPrefersDiverseDirectionFirst) {
  // Node 0 at origin.
  // Node 1 at (1.0, 0.0): closest, selected first.
  // Node 2 at (0.9, 1.5): ratio d(0,2)/d(1,2) ~= 1.163 -> occluded at cur_alpha=1.0,
  //   admitted at cur_alpha=1.2.
  // Node 3 at (-2.0, 0.0): opposite direction, ratio = 2/3 = 0.667 -> admitted at
  //   cur_alpha=1.0 even though farther than node 2.
  auto ds = makeDataset(2, {
      {0.0F, 0.0F},
      {1.0F, 0.0F},
      {0.9F, 1.5F},
      {-2.0F, 0.0F},
  });
  Graph graph(4, /*degreeThreshold=*/2);
  Vamana vamana(std::move(ds), std::move(graph), /*distanceThreshold=*/1.2F);
  vamana.setSaturateGraph(false);

  const std::vector<Neighbour> candidates = {
      {1.0F, 1},
      {3.06F, 2},
      {4.0F, 3},
  };
  vamana.prune(0, candidates);

  const NodeList& out = vamana.getOutNeighbors(0);
  ASSERT_EQ(out.size(), 2U);
  // Progressive cur_alpha=1.0 picks 1 and 3 (diverse opposite direction) before
  // relaxing to 1.2 where 2 would have taken slot 2 in a single-pass prune.
  EXPECT_EQ(out[0], 1U);
  EXPECT_EQ(out[1], 3U);
}

TEST(VamanaPruneTest, SaturateGraphBackfillsStarvedCollinearNodesUpToR) {
  // Collinear nodes 0..5 along +x. With alpha=1.0 and saturate=false, node 0
  // keeps only node 1 because node 1 occludes 2..5. With saturate=true, node 0
  // backfills the closest remaining candidates up to R=3.
  auto ds = makeDataset(1, {{0.0F}, {1.0F}, {2.0F}, {3.0F}, {4.0F}});
  Graph graph(5, /*degreeThreshold=*/3);
  Vamana vamana(std::move(ds), std::move(graph), /*distanceThreshold=*/1.0F);

  const std::vector<Neighbour> candidates = {
      {1.0F, 1}, {4.0F, 2}, {9.0F, 3}, {16.0F, 4},
  };

  vamana.setSaturateGraph(false);
  vamana.prune(0, candidates);
  EXPECT_EQ(vamana.getOutNeighbors(0), (NodeList{1}));

  vamana.setSaturateGraph(true);
  vamana.prune(0, candidates);
  EXPECT_EQ(vamana.getOutNeighbors(0), (NodeList{1, 2, 3}));
}

}  // namespace
