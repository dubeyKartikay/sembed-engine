#ifndef VAMANA
#define VAMANA
#include <filesystem>
#include <memory>
#include <vector>
#include "dataset.hpp"
#include "graph.hpp"
class Vamana{
  private:
  std::unique_ptr<DataSet> m_dataSet  ;
  Graph m_graph;
  float m_distanceThreshold;
  int m_degreeThreshold;
  void prune(int node, std::vector<int>& candidateSet );
  public:
  Vamana(std::unique_ptr<DataSet> dataSet);
  Vamana(Graph graph);
  Vamana(std::filesystem::path savedVamanaIndexPath);
  void setDistanceThreshold(float alpha);
  std::unique_ptr<std::vector<int>> search(int queryNode,int k);
  void save();
  ~Vamana();


};

#endif // !VAMANA



