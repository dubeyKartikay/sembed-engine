#ifndef VAMANA
#define VAMANA
#include "dataset.hpp"
#include "graph.hpp"
#include "searchresults.hpp"
#include <filesystem>
#include <memory>
#include <vector>
// make private fields private after testing
class Vamana {
public:
  std::unique_ptr<DataSet> m_dataSet;
  Graph m_graph;
  float m_distanceThreshold;
  int m_searchListSize;
  void prune(int node, std::vector<int> &candidateSet);
  SearchResults greedySearch(HDVector queryNode, int k);
  void insertIntoANN(std::vector<int> &NOut, std::vector<int> & ANNset, HDVector & nod );
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
  // take HDVector reference instead
  std::unique_ptr<std::vector<int>> search(HDVector queryNode, int k);
  void save();
/*   ~Vamana(); */
};

#endif // !VAMANA
