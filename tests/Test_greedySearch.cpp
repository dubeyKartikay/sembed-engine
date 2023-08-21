#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "vamana.hpp"
TEST(TestGreedySearch,TestingGreedySearchSimpleGraph){
    std::filesystem::path path("../build/testvec.bin");
  std::unique_ptr<InMemoryDataSet> dataset = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(dataset),2);
  v.setSeachListSize(100);
  HDVector hdve = *v.m_dataSet->getHDVecByIndex(2);
  SearchResults s = v.greedySearch( hdve, 1 ); 
  ASSERT_EQ(s.approximateNN.at(0), 2);
} 
