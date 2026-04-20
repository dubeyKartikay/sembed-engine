#include "graph.hpp"
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "utils.hpp"
Graph :: Graph(int64_t numberOfNodes, int64_t R){
  if (numberOfNodes < 0 || R < 0) {
    throw std::invalid_argument("graph sizes must be non-negative");
  }
  if (static_cast<uint64_t>(numberOfNodes) >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::overflow_error("graph node count exceeds addressable memory");
  }
  if (static_cast<uint64_t>(numberOfNodes) >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("graph node count exceeds int64_t range");
  }

  const uint64_t nodeCount = static_cast<uint64_t>(numberOfNodes);
  m_adj_list = std::vector<std::vector<int64_t>>(static_cast<size_t>(nodeCount),
                                                 std::vector<int64_t>());
  m_degreeThreshold = static_cast<uint64_t>(R);
  for (uint64_t i = 0; i < nodeCount; ++i) {
    m_adj_list[i] = generateRandomNumbers(m_degreeThreshold, numberOfNodes,i);
  }
  m_mediod = numberOfNodes == 0 ? -1 : getRandomNumber(0, numberOfNodes -1);
}

std::vector<int64_t> & Graph::getOutNeighbours(int64_t node){
  if (node < 0) {
    throw std::out_of_range("node index is outside graph bounds");
  }
  return m_adj_list.at(static_cast<size_t>(node)); 
}

void Graph::addOutNeighbourUnique(int64_t from, int64_t to) {
  if(to == from) return;
  if(from < 0 || to < 0 ||
     static_cast<uint64_t>(from) >= m_adj_list.size() ||
     static_cast<uint64_t>(to) >= m_adj_list.size()) {
    throw std::out_of_range("node index is outside graph bounds");
  }
  std::vector<int64_t> &neighbours = m_adj_list.at(static_cast<size_t>(from));
  if (std::find(neighbours.begin(), neighbours.end(), to) == neighbours.end()) {
    neighbours.push_back(to);
  }
}

void Graph::setOutNeighbours(int64_t node, const std::vector<int64_t> &neighbours) {
  if (node < 0) {
    throw std::out_of_range("node index is outside graph bounds");
  }
  auto &target = m_adj_list.at(static_cast<size_t>(node));
  target.clear();
  target.reserve(neighbours.size());
  for (int64_t neighbour : neighbours) {
    if (neighbour == node) {
      continue;
    }
    if (neighbour < 0 || static_cast<uint64_t>(neighbour) >= m_adj_list.size()) {
      throw std::out_of_range("node index is outside graph bounds");
    }
    target.push_back(neighbour);
  }
}

void Graph::clearOutNeighbours(int64_t node) {
  if (node < 0) {
    throw std::out_of_range("node index is outside graph bounds");
  }
  m_adj_list.at(static_cast<size_t>(node)).clear();
}
// todo- test
Graph::Graph(std::filesystem::path path){
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("could not open the graph file provided");
  }

  int64_t numberOfNodes = 0;
  int64_t degreeThreshold = 0;
  int64_t mediod = 0;
  file.read(reinterpret_cast<char*>(&numberOfNodes), sizeof(numberOfNodes));
  file.read(reinterpret_cast<char*>(&degreeThreshold), sizeof(degreeThreshold));
  file.read(reinterpret_cast<char*>(&mediod), sizeof(mediod));
  if (numberOfNodes < 0 || degreeThreshold < 0) {
    throw std::runtime_error("graph header contains negative values");
  }
  if (static_cast<uint64_t>(numberOfNodes) >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::overflow_error("graph node count exceeds addressable memory");
  }
  m_degreeThreshold = static_cast<uint64_t>(degreeThreshold);
  m_adj_list = std::vector<std::vector<int64_t>>(
      static_cast<size_t>(numberOfNodes),
      std::vector<int64_t>(static_cast<size_t>(m_degreeThreshold)));
  for (int64_t i = 0; i < numberOfNodes; ++i) {
    file.read(reinterpret_cast<char*>(m_adj_list[i].data()),
              static_cast<std::streamsize>(sizeof(int64_t) * degreeThreshold));
  }
  m_mediod = mediod;
}
