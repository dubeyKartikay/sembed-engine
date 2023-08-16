#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "vamana.hpp"
TEST(TestGreedySearch,TestingGreedySearchSimpleGraph){
    std::filesystem::path path("../build/gvec.bin");
  std::unique_ptr<InMemoryDataSet> dataset = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(dataset),50);
  HDVector hdve = *v.m_dataSet->getHDVecByIndex(0);
  SearchResults s = v.greedySearch( hdve, 10); 
  for (int ANN : s.approximateNN) {
    std::cout << ANN << std::endl;
  }
} 
