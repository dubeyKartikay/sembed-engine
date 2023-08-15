#include "vamana.hpp"
#include <algorithm>
#include <filesystem>
#include <memory>
Vamana::Vamana(std::unique_ptr<DataSet> datset, int degreeThreshold,float distanceThreshold) :m_graph(datset->getN(),degreeThreshold){
  m_dataSet = std::move(datset);
  m_distanceThreshold = distanceThreshold;
}
Vamana::Vamana(std::unique_ptr<DataSet> dataSet,Graph Graph,float distanceThreshold): m_graph(Graph) {
  m_dataSet = std::move(dataSet); 
  m_distanceThreshold = distanceThreshold;
}
Vamana::Vamana(std::unique_ptr<DataSet> dataSet, std::filesystem::path path,float distanceThreshold) : m_graph(path){
  m_dataSet = std::move(dataSet);
  m_distanceThreshold = distanceThreshold;
}

SearchResults Vamana::greedySearch(int node,int k){

}
