#include <cstdint>
#include <algorithm>
#include <memory>
#include <vector>
#include <filesystem>
#ifndef GRAPH
#define GRAPH
class Graph
{
private:
    std::vector<std::vector<int64_t>> m_adj_list;
    uint64_t m_degreeThreshold;
    int64_t m_mediod;
public:
    Graph() = default;
    
    Graph(int64_t numberOfNodes, int64_t R);
    Graph(std::filesystem::path path);
/*     ~Graph(); */
    std::vector<int64_t> & getOutNeighbours(int64_t node);
    void addOutNeighbourUnique(int64_t from, int64_t to);
    void setOutNeighbours(int64_t node, const std::vector<int64_t> &neighbours);
    void clearOutNeighbours(int64_t node);
    // void save();
    int64_t getMediod(){
      return m_mediod;
    }

    uint64_t getDegreeThreshold(){
      return m_degreeThreshold;
    }
};
#endif

// Graph::Graph(/* args */)
// {
// }

// Graph::~Graph()
// {
// }
