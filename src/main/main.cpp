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
  std::cout << "Loaded Dataset" << "N " << dataset->getN() << " D: "<< dataset->getDimentions() <<std::endl;
  Vamana v(std::move(dataset),2,1.2f);
  HDVector hdve = *v.m_dataSet->getHDVecByIndex(2);
  Graph g = v.m_graph;
  for (int neight : g.getOutNeighbours(2)) {
    std::cout << neight << " ";
  }
  std::cout << std::endl;
  std::cout << "mediod" << v.m_graph.getMediod() << std::endl;
  v.setSeachListSize(125);
  SearchResults s = v.greedySearch( hdve, 1); 
  for (int ANN : s.approximateNN) {
    std::cout << ANN << " " << HDVector::distance(hdve, *v.m_dataSet->getHDVecByIndex(ANN)) <<  std::endl;
  }
  std::cout<< "VISITDE" << std::endl;
  for (int vis : s.visited) {
    std::cout << vis << std::endl;
  }  
}
