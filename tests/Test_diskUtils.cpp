#include <gtest/gtest.h>
#include "dataset.hpp"
#include <filesystem>
#include <vector>
TEST(LoadingBinary, LoadingGLOVEBinaryInMemory) {
  std::filesystem::path path ("../build/gvec.bin");
  InMemoryDataSet dataset (path);
  ASSERT_EQ(dataset.getN(), 400000);
  ASSERT_EQ(dataset.getDimentions(), 51);
  std::vector<float> * data = new std::vector<float>({0,0.418 ,0.24968 ,-0.41242 ,0.1217 ,0.34527 ,-0.044457 ,-0.49688 ,-0.17862 ,-0.00066023 ,-0.6566 ,0.27843 ,-0.14767 ,-0.55677 ,0.14658 ,-0.0095095 ,0.011658 ,0.10204 ,-0.12792 ,-0.8443 ,-0.12181 ,-0.016801 ,-0.33279 ,-0.1552 ,-0.23131 ,-0.19181 ,-1.8823 ,-0.76746 ,0.099051 ,-0.42125 ,-0.19526 ,4.0071 ,-0.18594 ,-0.52287 ,-0.31681 ,0.00059213 ,0.0074449 ,0.17778 ,-0.15897 ,0.012041 ,-0.054223 ,-0.29871 ,-0.15749 ,-0.34758 ,-0.045637 ,-0.44251 ,0.18785 ,0.0027849,-0.18411,-0.11514,-0.78581});
  const HDVector & hdvec = dataset.getHDVecByIndex(0);
  for (int i = 0; i<dataset.getDimentions(); i++) {
     EXPECT_EQ(hdvec[i], data->at(i));
  }
  std::vector<float> * data2  = new std::vector<float>({ (float)(dataset.getN()-1),0.072617 ,-0.51393 ,0.4728 ,-0.52202 ,-0.35534 ,0.34629 ,0.23211 ,0.23096 ,0.26694 ,0.41028 ,0.28031 ,0.14107 ,-0.30212 ,-0.21095 ,-0.10875 ,-0.33659 ,-0.46313 ,-0.40999 ,0.32764 ,0.47401 ,-0.43449 ,0.19959 ,-0.55808 ,-0.34077 ,0.078477 ,0.62823 ,0.17161 ,-0.34454 ,-0.2066 ,0.1323 ,-1.8076 ,-0.38851 ,0.37654 ,-0.50422 ,-0.012446 ,0.046182 ,0.70028 ,-0.010573 ,-0.83629 ,-0.24698 ,0.6888 ,-0.17986 ,-0.066569 ,-0.48044 ,-0.55946 ,-0.27594 ,0.056072 ,-0.18907 ,-0.59021 ,0.55559});
  free(data);
  
  const HDVector & hdvec2 = dataset.getHDVecByIndex(dataset.getN()-1);
  for (int i = 0; i<dataset.getDimentions(); i++) {
    EXPECT_EQ(hdvec2[i], data2->at(i));
  }

} 
