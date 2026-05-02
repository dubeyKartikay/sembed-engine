#ifndef VAMANA
#define VAMANA

#include <cstdint>
#include <filesystem>
#include <memory>

#include "dataset.hpp"
#include "graph.hpp"
#include "node_types.hpp"
#include "searchresults.hpp"
#include "vector_view.hpp"
#include <boost/dynamic_bitset.hpp>

class Vamana {
public:
  void prune(NodeId node, NodeList &candidateSet);
  SearchResults greedySearch(FloatVectorView query, uint64_t k);
  void insertIntoSet(const NodeList &from, NodeList &to,
                     FloatVectorView comparisonVector);
  void insertIntoSet(const NodeList &from, SortedBoundedVector &to,
                     FloatVectorView comparisonVector, boost::dynamic_bitset<> &visited);
  Vamana(std::unique_ptr<DataSet> dataSet, uint64_t degreeThreshold,
         float distanceThreshold = 1.2f);
  Vamana(std::unique_ptr<DataSet> dataSet, Graph graph,
         float distanceThreshold = 1.2f);
  Vamana(std::unique_ptr<DataSet> dataSet,
         std::filesystem::path savedVamanaIndexPath,
         float distanceThreshold = 1.2f);

  float getDistanceThreshold() const { return m_distanceThreshold; }
  void setDistanceThreshold(float alpha) { m_distanceThreshold = alpha; }

  uint64_t getSearchListSize() const { return m_searchListSize; }
  bool isToBePruned(NodeId pDash, NodeId pStar, NodeId p);
  void setSearchListSize(int64_t searchListSize) {
    if (searchListSize < 0) {
      throw std::invalid_argument("search list size must be non-negative");
    }
    m_searchListSize = static_cast<uint64_t>(searchListSize);
  }

  uint64_t getDegreeThreshold() const { return m_graph.getDegreeThreshold(); }
  OptionalNodeId getMedoid() const { return m_graph.getMedoid(); }
  uint64_t getNodeCount() const { return m_graph.getNodeCount(); }
  const NodeList &getOutNeighbors(NodeId node) const {
    return m_graph.getOutNeighbors(node);
  }
  void setOutNeighbors(NodeId node, const NodeList &neighbors) {
    m_graph.setOutNeighbors(node, neighbors);
  }
  void clearOutNeighbors(NodeId node) {
    m_graph.clearOutNeighbors(node);
  }
  RecordView getRecordViewByIndex(NodeId node) const {
    return m_dataSet->getRecordViewByIndex(node);
  }

  void buildIndex();
  std::unique_ptr<NodeList> search(NodeId queryNode, uint64_t k);
  void save(std::filesystem::path path) const;

private:
  std::unique_ptr<DataSet> m_dataSet;
  Graph m_graph;
  float m_distanceThreshold = 1.2f;
  uint64_t m_searchListSize = 100;
};

#endif  // VAMANA
