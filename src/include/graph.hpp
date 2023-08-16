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
    int m_mediod;
public:
    Graph() = default;
    
    Graph(int numberOfNodes,int R);
    Graph(std::filesystem::path path);
/*     ~Graph(); */
    std::vector<int> & getOutNeighbours(const int& node);
    // void save();
    int getMediod(){
      return m_mediod;
    }
};
#endif

// Graph::Graph(/* args */)
// {
// }

// Graph::~Graph()
// {
// }
