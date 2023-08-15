#include "graph.hpp"
#include <fstream>
#include <vector>
#include "utils.hpp"
//todo -- test
Graph :: Graph(int numberOfNodes,int R){
  m_adj_list = std::vector<std::vector<int>>(numberOfNodes,std::vector<int>());
  m_degreeThreshold = R;
  for (int i = 0; i < numberOfNodes; i++) {
    m_adj_list[i] = generateRandomNumbers(m_degreeThreshold, numberOfNodes,i);
  }
  m_mediod = getRandomNumber(0, numberOfNodes -1);
}

std::vector<int> & Graph::getOutNeighbours(const int& node){
  return m_adj_list[node]; 
}
// todo- test
Graph::Graph(std::filesystem::path path){
  std::ifstream file(path);
  int numberOfNodes = 0;
  file.read(reinterpret_cast<char*>(&numberOfNodes), sizeof(int));
  file.read(reinterpret_cast<char*>(&m_degreeThreshold), sizeof(int));
   m_adj_list = std::vector<std::vector<int>>(numberOfNodes,std::vector<int>());
  for (size_t i = 0; i < numberOfNodes; i++) {
    file.read(reinterpret_cast<char*>(m_adj_list[i].data()), sizeof(int)*m_degreeThreshold);
  }
  m_mediod = getRandomNumber(0, numberOfNodes -1);
}
