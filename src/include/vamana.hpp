#ifndef VAMANA
#define VAMANA

#include <cstdint>
#include <filesystem>
#include <memory>

#include "dataset.hpp"
#include "graph.hpp"
#include "node_types.hpp"
#include "searchresults.hpp"
#include "utils.hpp"

class Vamana {
public:
  void prune(NodeId node, NodeList &candidateSet);
  SearchResults greedySearch(const HDVector &queryNode, uint64_t k);
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

  float getDistanceThreshold() const { return distanceThreshold_; }
  void setDistanceThreshold(float alpha) { distanceThreshold_ = alpha; }

  uint64_t getSearchListSize() const { return searchListSize_; }
  bool isToBePruned(NodeId p_dash, NodeId p_start, NodeId p);
  void setSearchListSize(int64_t searchListSize) {
    if (searchListSize < 0) {
      throw std::invalid_argument("search list size must be non-negative");
    }
    searchListSize_ = static_cast<uint64_t>(searchListSize);
  }

  uint64_t getDegreeThreshold() const { return graph_.getDegreeThreshold(); }
  OptionalNodeId getMedoid() const { return graph_.getMedoid(); }
  uint64_t getNodeCount() const { return graph_.getNodeCount(); }
  const NodeList &getOutNeighbors(NodeId node) const {
    return graph_.getOutNeighbors(node);
  }
  void setOutNeighbors(NodeId node, const NodeList &neighbors) {
    graph_.setOutNeighbors(node, neighbors);
  }
  void clearOutNeighbors(NodeId node) {
    graph_.clearOutNeighbors(node);
  }
  RecordView getRecordViewByIndex(NodeId node) const {
    return dataSet_->getRecordViewByIndex(node);
  }

  void buildIndex();
  std::unique_ptr<NodeList> search(NodeId queryNode, uint64_t k);
  void save(std::filesystem::path path) const;

private:
  std::unique_ptr<DataSet> dataSet_;
  Graph graph_;
  float distanceThreshold_ = 1.2f;
  uint64_t searchListSize_ = 100;
};

#endif // !VAMANA
