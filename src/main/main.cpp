#include<iostream>
#include <cstdint>
#include "load_from_binary.hpp"
#include <filesystem>
#include <iterator>
#include <string>
#include "dataset.hpp"
#include "node_types.hpp"
#include "vamana.hpp"
int main(int argc, char ** argv){
    (void)argc;
    (void)argv;
    // std::cout << argc << argv[0] << '\n';
  std::filesystem::path path("../build/gvec.bin");
  std::unique_ptr<InMemoryDataSet> dataset = std::make_unique<InMemoryDataSet>(path);
  std::cout << "Loaded Dataset" << "N " << dataset->getN() << " D: "<< dataset->getDimentions() <<std::endl;
  Vamana v(std::move(dataset),70,1.2f);
  RecordView queryRecord = v.m_dataSet->getRecordViewByIndex(2);
  HDVector hdve = *queryRecord.vector;
  Graph g = v.m_graph;
  for (NodeId neight : g.getOutNeighbours(2)) {
    std::cout << neight << " ";
  }
  std::cout << std::endl;
  if (const OptionalNodeId mediod = v.m_graph.getMediod()) {
    std::cout << "mediod" << *mediod << std::endl;
  } else {
    std::cout << "mediod none" << std::endl;
  }
  v.setSeachListSize(125);
  SearchResults s = v.greedySearch( hdve, 1); 
  for (NodeId ANN : s.approximateNN) {
    RecordView annRecord = v.m_dataSet->getRecordViewByIndex(ANN);
    std::cout << annRecord.recordId << " "
              << HDVector::distance(hdve, *annRecord.vector) <<  std::endl;
  }
  std::cout<< "VISITDE" << std::endl;
  for (NodeId vis : s.visited) {
    std::cout << vis << std::endl;
  }  
}
