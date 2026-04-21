#ifndef GRAPH
#define GRAPH

#include <filesystem>

#include "node_types.hpp"

class Vamana;

class Graph {
public:
  Graph() = default;
  Graph(NodeId numberOfNodes, uint64_t degreeThreshold);
  Graph(std::filesystem::path path);

  const NodeList &getOutNeighbors(NodeId node) const;
  void addOutNeighborUnique(NodeId from, NodeId to);
  void setOutNeighbors(NodeId node, const NodeList &neighbors);
  void clearOutNeighbors(NodeId node);

  OptionalNodeId getMedoid() const { return m_medoid; }
  uint64_t getDegreeThreshold() const { return m_degreeThreshold; }
  uint64_t getNodeCount() const { return static_cast<uint64_t>(m_adjList.size()); }

  void save(std::filesystem::path path) const;

private:
  NodeList &mutableOutNeighbors(NodeId node);

  std::vector<NodeList> m_adjList;
  uint64_t m_degreeThreshold = 0;
  OptionalNodeId m_medoid;

  friend class Vamana;
};

#endif  // GRAPH
