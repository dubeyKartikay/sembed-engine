#include <filesystem>
#include "node_types.hpp"
#ifndef GRAPH
#define GRAPH
class Graph
{
private:
    std::vector<NodeList> m_adj_list;
    uint64_t m_degreeThreshold;
    OptionalNodeId m_mediod;
public:
    Graph() = default;
    Graph(NodeId numberOfNodes, uint64_t R);
    Graph(std::filesystem::path path);
    NodeList & getOutNeighbours(NodeId node);
    void addOutNeighbourUnique(NodeId from, NodeId to);
    void setOutNeighbours(NodeId node, const NodeList &neighbours);
    void clearOutNeighbours(NodeId node);
    OptionalNodeId getMediod() const {
      return m_mediod;
    }
    uint64_t getDegreeThreshold() const {
      return m_degreeThreshold;
    }
    void save(std::filesystem::path path);
};
#endif
