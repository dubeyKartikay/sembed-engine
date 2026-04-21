#ifndef VAMANA
#define VAMANA
#include <cstdint>
#include "dataset.hpp"
#include "graph.hpp"
#include "node_types.hpp"
#include "searchresults.hpp"
#include <filesystem>
#include <memory>
#include <vector>
#include "utils.hpp"
// make private fields private after testing
class Vamana {
public:
  std::unique_ptr<DataSet> m_dataSet;
  Graph m_graph;
  float m_distanceThreshold;
  uint64_t m_searchListSize;
  void prune(NodeId node, NodeList &candidateSet);
  SearchResults greedySearch(const HDVector & queryNode, uint64_t k);
  void insertIntoSet(const NodeList &NOut,
                     NodeList &ANNset,
                     const HDVector &nod);
  Vamana(std::unique_ptr<DataSet> dataSet, uint64_t degreeThreshold,
         float distanceThreshold = 1.2f);
  Vamana(std::unique_ptr<DataSet> dataSet, Graph graph,
         float distanceThreshold = 1.2f);
  Vamana(std::unique_ptr<DataSet> dataSet,
         std::filesystem::path savedVamanaIndexPath,
         float distanceThreshold = 1.2f);
  void setDistanceThreshold(float alpha) { m_distanceThreshold = alpha; };
  bool isToBePruned(NodeId p_dash, NodeId p_start, NodeId p);
  void setSeachListSize(int64_t L) {
    if (L < 0) {
      throw std::invalid_argument("search list size must be non-negative");
    }
    m_searchListSize = static_cast<uint64_t>(L);
  }
  // take HDVector reference instead
  void buildIndex();
  std::unique_ptr<NodeList> search(NodeId queryNode, uint64_t k);
  void save(std::filesystem::path path);
  /*   ~Vamana(); */
};

#endif // !VAMANA
