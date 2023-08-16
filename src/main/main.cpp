#include<iostream>
#include "load_from_binary.hpp"
#include <filesystem>
#include <iterator>
#include <string>
#include "dataset.hpp"
#include "vamana.hpp"
int main(int argc, char ** argv){
    // std::cout << argc << argv[0] << '\n';
  std::filesystem::path path("../build/testvec.bin");
  std::unique_ptr<InMemoryDataSet> dataset = std::make_unique<InMemoryDataSet>(path);
  Vamana v(std::move(dataset),500);
  HDVector hdve = *v.m_dataSet->getHDVecByIndex(2);
  Graph g = v.m_graph;
  for (int i = 0; i < v.m_dataSet->getN(); i++) {
    for (int ele : g.getOutNeighbours(i)) {
      std::cout << ele << " "; 
    }
    std::cout << std::endl;
  }
  v.setSeachListSize(100);
  SearchResults s = v.greedySearch( hdve, 10); 
  for (int ANN : s.approximateNN) {
    std::cout << ANN << " " << HDVector::distance(hdve, *v.m_dataSet->getHDVecByIndex(ANN)) <<  std::endl;
  }
  std::cout<< "VISITDE" << std::endl;
  for (int vis : s.visited) {
    std::cout << vis << std::endl;
  }  
}
