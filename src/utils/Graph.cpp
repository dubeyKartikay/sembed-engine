#include "graph.hpp"
#include <vector>
#include "utils.hpp"
Graph :: Graph(int numberOfNodes,int R){
  m_adj_list = std::vector<std::vector<int>>(numberOfNodes,std::vector<int>());
  m_degreeThreshold = R;
  for (int i = 0; i < numberOfNodes; i++) {
    m_adj_list[i] = generateRandomNumbers(m_degreeThreshold, numberOfNodes,i);
  }
}

std::vector<int> & Graph::getOutNeighbours(const int& node){
  return m_adj_list[node]; 
}
