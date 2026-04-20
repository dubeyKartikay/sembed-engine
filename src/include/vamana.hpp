#ifndef VAMANA
#define VAMANA
#include <cstdint>
#include "dataset.hpp"
#include "graph.hpp"
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
  void prune(int64_t node, std::vector<int64_t> &candidateSet);
  SearchResults greedySearch(HDVector & queryNode, uint64_t k);
  void insertIntoSet(const std::vector<int64_t> &NOut,
                     std::vector<int64_t> &ANNset,
                     HDVector &nod);
  Vamana(std::unique_ptr<DataSet> dataSet, int64_t degreeThreshold,
         float distanceThreshold = 1.2f);
  Vamana(std::unique_ptr<DataSet> dataSet, Graph graph,
         float distanceThreshold = 1.2f);
  Vamana(std::unique_ptr<DataSet> dataSet,
         std::filesystem::path savedVamanaIndexPath,
         float distanceThreshold = 1.2f);
  void setDistanceThreshold(float alpha) { m_distanceThreshold = alpha; };
  bool isToBePruned(int64_t p_dash, int64_t p_start, int64_t p);
  void setSeachListSize(int64_t L) {
    if (L < 0) {
      throw std::invalid_argument("search list size must be non-negative");
    }
    m_searchListSize = static_cast<uint64_t>(L);
  }
  // take HDVector reference instead
  void buildIndex();
  std::unique_ptr<std::vector<int64_t>> search(HDVector queryNode, uint64_t k);
  void save();
  /*   ~Vamana(); */
};

#endif // !VAMANA
