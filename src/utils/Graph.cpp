#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "graph.hpp"
#include "utils.hpp"

namespace {

void validateNeighborList(const NodeList &neighbors, NodeId node,
                          uint64_t nodeCount, uint64_t degreeThreshold) {
  if (neighbors.size() > degreeThreshold) {
    throw std::invalid_argument("graph adjacency exceeds degree threshold");
  }

  std::unordered_set<NodeId> seen;
  for (NodeId neighbor : neighbors) {
    if (neighbor == node) {
      throw std::invalid_argument("graph adjacency contains self-loop");
    }
    if (neighbor >= nodeCount) {
      throw std::out_of_range("node index is outside graph bounds");
    }
    if (!seen.insert(neighbor).second) {
      throw std::invalid_argument("graph adjacency contains duplicate node id");
    }
  }
}

}  // namespace

Graph::Graph(NodeId numberOfNodes, uint64_t degreeThreshold) {
  if (numberOfNodes >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::overflow_error("graph node count exceeds addressable memory");
  }
  m_adjList =
      std::vector<NodeList>(static_cast<size_t>(numberOfNodes), NodeList());
  m_degreeThreshold = degreeThreshold;
  #pragma omp parallel for
  for (NodeId i = 0; i < numberOfNodes; ++i) {
    auto rng = makeDeterministicRng(0x6f72617068536565ULL,
                                    {numberOfNodes, degreeThreshold, i});
    m_adjList[static_cast<size_t>(i)] =
        generateRandomNumbers(m_degreeThreshold, numberOfNodes, rng, i);
  }

  m_medoid = numberOfNodes == 0 ? std::nullopt
                                : OptionalNodeId(numberOfNodes / 2);
}

NodeList &Graph::mutableOutNeighbors(NodeId node) {
  if (node >= static_cast<uint64_t>(m_adjList.size())) {
    throw std::out_of_range("node index is outside graph bounds");
  }
  return m_adjList.at(static_cast<size_t>(node));
}

const NodeList &Graph::getOutNeighbors(NodeId node) const {
  if (node >= static_cast<uint64_t>(m_adjList.size())) {
    throw std::out_of_range("node index is outside graph bounds");
  }
  return m_adjList.at(static_cast<size_t>(node));
}

void Graph::addOutNeighborUnique(NodeId from, NodeId to) {
  if (to == from) {
    return;
  }
  if (to >= static_cast<uint64_t>(m_adjList.size())) {
    throw std::out_of_range("node index is outside graph bounds");
  }

  NodeList &neighbors = mutableOutNeighbors(from);
  if (std::find(neighbors.begin(), neighbors.end(), to) == neighbors.end()) {
    neighbors.push_back(to);
  }
}

void Graph::setOutNeighbors(NodeId node, const NodeList &neighbors) {
  if (node >= static_cast<uint64_t>(m_adjList.size())) {
    throw std::out_of_range("node index is outside graph bounds");
  }

  validateNeighborList(neighbors, node,
                       static_cast<uint64_t>(m_adjList.size()),
                       m_degreeThreshold);
  NodeList &target = mutableOutNeighbors(node);
  target = neighbors;
}

void Graph::clearOutNeighbors(NodeId node) {
  if (node >= static_cast<uint64_t>(m_adjList.size())) {
    throw std::out_of_range("node index is outside graph bounds");
  }
  m_adjList.at(static_cast<size_t>(node)).clear();
}

Graph::Graph(std::filesystem::path path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("could not open the graph file provided");
  }

  uint64_t numberOfNodes = 0;
  uint64_t degreeThreshold = 0;
  uint64_t medoid = std::numeric_limits<uint64_t>::max();
  file.read(reinterpret_cast<char *>(&numberOfNodes), sizeof(numberOfNodes));
  file.read(reinterpret_cast<char *>(&degreeThreshold), sizeof(degreeThreshold));
  file.read(reinterpret_cast<char *>(&medoid), sizeof(medoid));
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
  m_adjList = std::vector<NodeList>(static_cast<size_t>(numberOfNodes));
  for (size_t i = 0; i < numberOfNodes; ++i) {
    auto &nodeAdjacency = m_adjList[i];
    uint64_t degree = 0;
    file.read(reinterpret_cast<char *>(&degree), sizeof(degree));
    if (!file) {
      throw std::runtime_error("failed to read graph adjacency");
    }
    if (degree > m_degreeThreshold) {
      throw std::runtime_error("graph adjacency exceeds degree threshold");
    }
    nodeAdjacency.resize(static_cast<size_t>(degree));
    file.read(reinterpret_cast<char *>(nodeAdjacency.data()),
              static_cast<std::streamsize>(sizeof(NodeId) * degree));
    if (!file) {
      throw std::runtime_error("failed to read graph adjacency");
    }
    try {
      validateNeighborList(nodeAdjacency, static_cast<NodeId>(i), numberOfNodes,
                           m_degreeThreshold);
    } catch (const std::out_of_range &) {
      throw std::runtime_error("graph adjacency contains invalid node id");
    } catch (const std::invalid_argument &error) {
      throw std::runtime_error(error.what());
    }
  }

  if (medoid == std::numeric_limits<uint64_t>::max()) {
    m_medoid = std::nullopt;
  } else if (medoid < numberOfNodes) {
    m_medoid = medoid;
  } else {
    throw std::runtime_error("graph header contains invalid medoid");
  }

  char extra = '\0';
  if (file.read(&extra, 1)) {
    throw std::runtime_error("graph file contains trailing data");
  }
  if (!file.eof()) {
    throw std::runtime_error("failed to validate graph file size");
  }
}

void Graph::save(std::filesystem::path path) const {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("could not open the graph file provided");
  }

  uint64_t numberOfNodes = static_cast<uint64_t>(m_adjList.size());
  uint64_t degreeThreshold = m_degreeThreshold;
  if (m_medoid && *m_medoid >= numberOfNodes) {
    throw std::runtime_error("graph header contains invalid medoid");
  }
  uint64_t medoid = m_medoid.value_or(std::numeric_limits<uint64_t>::max());
  file.write(reinterpret_cast<const char *>(&numberOfNodes),
             sizeof(numberOfNodes));
  file.write(reinterpret_cast<const char *>(&degreeThreshold),
             sizeof(degreeThreshold));
  file.write(reinterpret_cast<const char *>(&medoid), sizeof(medoid));
  if (!file) {
    throw std::runtime_error("failed to write graph header");
  }

  for (size_t i = 0; i < numberOfNodes; ++i) {
    const auto &nodeAdjacency = m_adjList[i];
    validateNeighborList(nodeAdjacency, static_cast<NodeId>(i), numberOfNodes,
                         m_degreeThreshold);

    uint64_t degree = static_cast<uint64_t>(nodeAdjacency.size());
    file.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
    file.write(reinterpret_cast<const char *>(nodeAdjacency.data()),
              static_cast<std::streamsize>(sizeof(NodeId) * degree));
    if (!file) {
      throw std::runtime_error("failed to write graph adjacency");
    }
  }
}
