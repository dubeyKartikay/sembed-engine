#include "ArmaVector.hpp"
#include "HDVector.hpp"
#include "vector_view.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

TEST(VectorCore, ComputesEuclideanDistanceForTwoDimensions) {
  HDVector left(std::vector<float>{0.0f, 0.0f});
  HDVector right(std::vector<float>{3.0f, 4.0f});

  ASSERT_FLOAT_EQ(euclideanDistance(left.view(), right.view()), 5.0f);
}

TEST(VectorCore, ComputesEuclideanDistanceForFourDimensions) {
  HDVector left(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  HDVector right(std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f});

  ASSERT_FLOAT_EQ(euclideanDistance(left.view(), right.view()), 8.0f);
}

TEST(VectorCore, ComputesSquaredDistanceWithoutSquareRoot) {
  HDVector left(std::vector<float>{0.0f, 0.0f});
  HDVector right(std::vector<float>{3.0f, 4.0f});

  ASSERT_FLOAT_EQ(squaredDistance(left.view(), right.view()), 25.0f);
}

TEST(VectorCore, HDVectorAndArmaVectorInteroperateThroughViews) {
  HDVector left(std::vector<float>{1.0f, 2.0f});
  ArmaVector right(std::vector<float>{4.0f, 6.0f});

  ASSERT_FLOAT_EQ(euclideanDistance(left.view(), right.view()), 5.0f);
  ASSERT_FLOAT_EQ(euclideanDistance(right.view(), left.view()), 5.0f);
}

TEST(VectorCore, DistanceRejectsMismatchedDimensions) {
  HDVector left(std::vector<float>{1.0f, 2.0f});
  HDVector right(std::vector<float>{1.0f});

  EXPECT_THROW((void)squaredDistance(left.view(), right.view()),
               std::invalid_argument);
}
