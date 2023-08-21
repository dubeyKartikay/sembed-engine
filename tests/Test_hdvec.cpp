#include <gtest/gtest.h>
#include <vector>
#include "HDVector.hpp"
TEST(TestHDVector,TestinHDVectorDistanceSimple){
  std::vector<float> v1 = {0,0.0f,0.0f};
  std::vector<float> v2 = {1,3.0f,4.0f};
  HDVector hv1(v1);
  HDVector hv2(v2);
  ASSERT_FLOAT_EQ(HDVector::distance(v1, v2), 5.0f);
}

TEST(TestHDVector,TestinHDVectorDistanceNormal){
  std::vector<float> v1 = {0,1,2,3,4};
  
  std::vector<float> v2 = {1,5,6,7,8};
  HDVector hv1(v1);
  HDVector hv2(v2);
  ASSERT_FLOAT_EQ(HDVector::distance(hv1, hv2), 8.0f);
}
