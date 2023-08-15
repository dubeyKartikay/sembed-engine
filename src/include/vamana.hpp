#ifndef VAMANA
#define VAMANA
#include "dataset.hpp"
#include "graph.hpp"
#include "searchresults.hpp"
#include <filesystem>
#include <memory>
#include <vector>
class Vamana {
private:
  std::unique_ptr<DataSet> m_dataSet;
  Graph m_graph;
  float m_distanceThreshold;
  int m_searchListSize;
  void prune(int node, std::vector<int> &candidateSet);
  SearchResults greedySearch(int queryNode, int k);

public:
  Vamana(std::unique_ptr<DataSet> dataSet, int degreeThreshold,
         float distanceThreshold = 1.2f);
  Vamana(std::unique_ptr<DataSet> dataSet, Graph graph,
         float distanceThreshold = 1.2f);
  Vamana(std::unique_ptr<DataSet> dataSet,
         std::filesystem::path savedVamanaIndexPath,
         float distanceThreshold = 1.2f);
  void setDistanceThreshold(float alpha){
    m_distanceThreshold = alpha;
  };
  void setSeachListSize(int L){
    m_searchListSize = L;
  }
  std::unique_ptr<std::vector<int>> search(int queryNode, int k);
  void save();
  ~Vamana();
};

#endif // !VAMANA
