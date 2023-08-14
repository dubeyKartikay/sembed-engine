#include <algorithm>
#include <memory>
#include <vector>
#include <filesystem>
#ifndef GRAPH
#define GRAPH
class Graph
{
private:
    std::vector<std::vector<int>> m_adj_list;
    int m_degreeThreshold;
public:
    Graph(int numberOfNodes,int R);
    ~Graph();
    std::vector<int> & getOutNeighbours(const int& node);
    // void save();
    // void load(std::filesystem::path path);
};
#endif

// Graph::Graph(/* args */)
// {
// }

// Graph::~Graph()
// {
// }
