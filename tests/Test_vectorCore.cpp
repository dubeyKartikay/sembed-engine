#include "HDVector.hpp"
#include "Vector.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <type_traits>
#include <vector>

static_assert(std::is_abstract_v<Vector>,
              "Vector must remain an abstract interface");
static_assert(std::is_base_of_v<Vector, HDVector>,
              "HDVector must implement Vector");

TEST(VectorCore, ComputesEuclideanDistanceForTwoDimensions) {
  HDVector left(std::vector<float>{0.0f, 0.0f});
  HDVector right(std::vector<float>{3.0f, 4.0f});

  ASSERT_FLOAT_EQ(Vector::distance(left, right), 5.0f);
}

TEST(VectorCore, ComputesEuclideanDistanceForFourDimensions) {
  HDVector left(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  HDVector right(std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f});

  ASSERT_FLOAT_EQ(Vector::distance(left, right), 8.0f);
}

TEST(VectorCore, SharedPtrToHDVectorUpcastsToVector) {
  std::shared_ptr<Vector> vector =
      std::make_shared<HDVector>(std::vector<float>{7.0f, 8.0f});

  ASSERT_NE(vector, nullptr);
  EXPECT_EQ(vector->getDimension(), 2U);
  EXPECT_FLOAT_EQ((*vector)[0], 7.0f);
  EXPECT_FLOAT_EQ((*vector)[1], 8.0f);
}
