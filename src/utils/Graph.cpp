#include "graph.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include "utils.hpp"

namespace {
NodeId randomNodeId(NodeId start, NodeId end) {
  std::random_device device;
  std::mt19937_64 generator(device());
  std::uniform_int_distribution<NodeId> distribution(start, end);
  return distribution(generator);
}
}  // namespace

Graph::Graph(NodeId numberOfNodes, uint64_t R) {
  if (numberOfNodes >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::overflow_error("graph node count exceeds addressable memory");
  }
  m_adj_list =
      std::vector<NodeList>(static_cast<size_t>(numberOfNodes), NodeList());
  m_degreeThreshold = static_cast<uint64_t>(R);
  for (NodeId i = 0; i < numberOfNodes; ++i) {
    m_adj_list[static_cast<size_t>(i)] =
        generateRandomNumbers(m_degreeThreshold, numberOfNodes, i);
  }
  m_mediod = numberOfNodes == 0 ? std::nullopt
                                : OptionalNodeId(randomNodeId(0, numberOfNodes - 1));
}

NodeList &Graph::getOutNeighbours(NodeId node) {
  if (node >= static_cast<uint64_t>(m_adj_list.size())) {
    throw std::out_of_range("node index is outside graph bounds");
  }
  return m_adj_list.at(static_cast<size_t>(node));
}

void Graph::addOutNeighbourUnique(NodeId from, NodeId to) {
  if (to == from) return;
  if (from >= static_cast<uint64_t>(m_adj_list.size()) ||
      to >= static_cast<uint64_t>(m_adj_list.size())) {
    throw std::out_of_range("node index is outside graph bounds");
  }
  NodeList &neighbours = m_adj_list.at(static_cast<size_t>(from));
  if (std::find(neighbours.begin(), neighbours.end(), to) == neighbours.end()) {
    neighbours.push_back(to);
  }
}

void Graph::setOutNeighbours(NodeId node, const NodeList &neighbours) {
  if (node >= static_cast<uint64_t>(m_adj_list.size())) {
    throw std::out_of_range("node index is outside graph bounds");
  }
  auto &target = m_adj_list.at(static_cast<size_t>(node));
  target.clear();
  target.reserve(neighbours.size());
  for (NodeId neighbour : neighbours) {
    if (neighbour == node) {
      continue;
    }
    if (neighbour >= static_cast<uint64_t>(m_adj_list.size())) {
      throw std::out_of_range("node index is outside graph bounds");
    }
    target.push_back(neighbour);
  }
}

void Graph::clearOutNeighbours(NodeId node) {
  if (node >= static_cast<uint64_t>(m_adj_list.size())) {
    throw std::out_of_range("node index is outside graph bounds");
  }
  m_adj_list.at(static_cast<size_t>(node)).clear();
}

Graph::Graph(std::filesystem::path path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("could not open the graph file provided");
  }

  uint64_t numberOfNodes = 0;
  uint64_t degreeThreshold = 0;
  uint64_t mediod = std::numeric_limits<uint64_t>::max();
  file.read(reinterpret_cast<char *>(&numberOfNodes), sizeof(numberOfNodes));
  file.read(reinterpret_cast<char *>(&degreeThreshold), sizeof(degreeThreshold));
  file.read(reinterpret_cast<char *>(&mediod), sizeof(mediod));
  if (!file) {
    throw std::runtime_error("failed to read graph header");
  }

  if (numberOfNodes >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::overflow_error("graph node count exceeds addressable memory");
  }
  if (degreeThreshold >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::overflow_error("graph degree exceeds addressable memory");
  }

  m_degreeThreshold = degreeThreshold;
  m_adj_list = std::vector<NodeList>(
      static_cast<size_t>(numberOfNodes),
      NodeList(static_cast<size_t>(m_degreeThreshold)));
  for (size_t i = 0; i < static_cast<size_t>(numberOfNodes); ++i) {
    file.read(reinterpret_cast<char *>(m_adj_list[i].data()),
              static_cast<std::streamsize>(sizeof(NodeId) * degreeThreshold));
    if (!file) {
      throw std::runtime_error("failed to read graph adjacency");
    }
    for (NodeId neighbour : m_adj_list[i]) {
      if (neighbour >= numberOfNodes) {
        throw std::runtime_error("graph adjacency contains invalid node id");
      }
    }
  }

  if (mediod == std::numeric_limits<uint64_t>::max()) {
    m_mediod = std::nullopt;
  } else if (mediod < numberOfNodes) {
    m_mediod = mediod;
  } else {
    throw std::runtime_error("graph header contains invalid mediod");
  }
}
