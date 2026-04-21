#include "vamana.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <unordered_set>
#include <vector>

namespace {
uint64_t checkedDatasetNodeCount(const std::unique_ptr<DataSet> &dataSet) {
  if (!dataSet) {
    throw std::invalid_argument("dataset must not be null");
  }
  return dataSet->getN();
}

void validateDataset(const std::unique_ptr<DataSet> &dataSet) {
  if (!dataSet) {
    throw std::invalid_argument("dataset must not be null");
  }
}
}  // namespace

Vamana::Vamana(std::unique_ptr<DataSet> datset, uint64_t degreeThreshold,
               float distanceThreshold)
    : graph_([&]() -> Graph {
        return Graph(checkedDatasetNodeCount(datset), degreeThreshold);
      }()) {
  validateDataset(datset);
  dataSet_ = std::move(datset);
  distanceThreshold_ = distanceThreshold;
  searchListSize_ = 100;
  buildIndex();
}
Vamana::Vamana(std::unique_ptr<DataSet> dataSet, Graph graph,
               float distanceThreshold)
    : graph_(std::move(graph)) {
  validateDataset(dataSet);
  dataSet_ = std::move(dataSet);
  distanceThreshold_ = distanceThreshold;
  searchListSize_ = 100;
}
Vamana::Vamana(std::unique_ptr<DataSet> dataSet, std::filesystem::path path,
               float distanceThreshold)
    : graph_(path) {
  validateDataset(dataSet);
  dataSet_ = std::move(dataSet);
  distanceThreshold_ = distanceThreshold;
  searchListSize_ = 100;
}
struct customLess {
    const HDVector *qNode;
    const Vamana *vamana;
    customLess(const HDVector *qNode, const Vamana *vamana)
        : qNode(qNode), vamana(vamana){};
    bool operator()(NodeId l, NodeId r) {
      const float leftDistance =
          HDVector::distance(*qNode, *(vamana->getRecordViewByIndex(l).vector));
      const float rightDistance =
          HDVector::distance(*qNode, *(vamana->getRecordViewByIndex(r).vector));
      if (leftDistance == rightDistance) {
        return l < r;
      }
      return leftDistance < rightDistance;
    }
  };
void Vamana::insertIntoSet(const NodeList &from,
                           NodeList &to,
                           const HDVector &comparison_vec) {
  for (const NodeId outNode : from) {

    auto pos = std::lower_bound(to.begin(), to.end(), outNode,
                                customLess(&comparison_vec, this));
    if (pos == to.end()) {
      to.push_back(outNode);
    } else {
      if (*pos != outNode) {
        to.insert(pos, outNode);
      }
    }
  }
}

SearchResults Vamana::greedySearch(const HDVector &node, uint64_t k) {
  SearchResults searchResult;
  const OptionalNodeId medoid = graph_.getMedoid();
  if (!medoid || searchListSize_ == 0 || k == 0) {
    return searchResult;
  }
  searchResult.approximateNN.push_back(*medoid);
  std::unordered_set<NodeId> visited;
  uint64_t maxIter = 0;
  while (maxIter < 10000) {
    uint64_t i = 0;
    while (i < searchListSize_ && i < searchResult.approximateNN.size() &&
           visited.count(searchResult.approximateNN[i]) != 0) {
      ++i;
    }
    if (i >= searchListSize_ || i >= searchResult.approximateNN.size()) {
      break;
    }

    const NodeId node_p_star = searchResult.approximateNN[i];
    visited.insert(node_p_star);
    searchResult.visited.push_back(node_p_star);
    insertIntoSet(graph_.getOutNeighbors(node_p_star),
                  searchResult.approximateNN, node);
    while (searchResult.approximateNN.size() > searchListSize_) {
      searchResult.approximateNN.pop_back();
    }
    maxIter++;
  }
  while (searchResult.approximateNN.size() > k) {
    searchResult.approximateNN.pop_back();
  }

  return searchResult;
}

bool Vamana::isToBePruned(NodeId p_dash, NodeId p_star, NodeId p) {
  std::shared_ptr<HDVector> p_dash_vec = dataSet_->getRecordViewByIndex(p_dash).vector;
  std::shared_ptr<HDVector> p_star_vec = dataSet_->getRecordViewByIndex(p_star).vector;
  std::shared_ptr<HDVector> p_vec = dataSet_->getRecordViewByIndex(p).vector;
  float distanceFromP_starToP_dash =
      HDVector::distance(*p_star_vec, *p_dash_vec);
  float distanceFromPToP_dash = HDVector::distance(*p_vec, *p_dash_vec);
  return distanceThreshold_ * distanceFromP_starToP_dash <=
         distanceFromPToP_dash;
}

void Vamana::prune(NodeId node, NodeList &candidateSet) {
  std::shared_ptr<HDVector> p_vec = dataSet_->getRecordViewByIndex(node).vector;
  const NodeList &outNeighbors = graph_.getOutNeighbors(node);
  NodeList candidates = candidateSet;
  insertIntoSet(outNeighbors, candidates, *p_vec);
  candidates.erase(std::remove(candidates.begin(), candidates.end(), node),
                   candidates.end());
  std::sort(candidates.begin(), candidates.end(),
            customLess(&*p_vec, this));
  graph_.clearOutNeighbors(node);
  if (graph_.getDegreeThreshold() == 0) {
    candidateSet.clear();
    return;
  }
  std::unordered_set<NodeId> inOutNeighbours;
  while (candidates.size() > 0) {
    const NodeId p_star = candidates[0];
    if (inOutNeighbours.count(p_star) == 0) {
      graph_.addOutNeighborUnique(node, p_star);
      inOutNeighbours.insert(p_star);
    }
    if (graph_.getOutNeighbors(node).size() == graph_.getDegreeThreshold()) {
      break;
    }

    auto it = candidates.begin();
    while (it != candidates.end()) {
      const NodeId p_dash = *it;
      if (isToBePruned(p_dash, p_star, node)) {
        it = candidates.erase(it);

      } else {
        it++;
      }
    }
  }
  if (&candidateSet != &graph_.mutableOutNeighbors(node)) {
    candidateSet = candidates;
  }
}

void Vamana::buildIndex() {
  graph_ = Graph(dataSet_->getN(), graph_.getDegreeThreshold());
  auto rng = makeDeterministicRng(
      0x76616d616e61524eULL,
      {dataSet_->getN(), graph_.getDegreeThreshold()},
      {distanceThreshold_});
  NodeList sigma = getPermutation(static_cast<int64_t>(dataSet_->getN()), rng);
  for (NodeId &node : sigma) {
    SearchResults greedySearchResult =
        greedySearch(*dataSet_->getRecordViewByIndex(node).vector, 1);
    prune(node, greedySearchResult.visited);
    for (NodeId neighbour : graph_.getOutNeighbors(node)) {
      try {
        graph_.addOutNeighborUnique(neighbour, node);
      } catch (const std::invalid_argument &) {
        prune(neighbour, graph_.mutableOutNeighbors(neighbour));
        if (std::find(graph_.getOutNeighbors(neighbour).begin(),
                      graph_.getOutNeighbors(neighbour).end(),
                      node) == graph_.getOutNeighbors(neighbour).end()) {
          if (graph_.getOutNeighbors(neighbour).size() <
              graph_.getDegreeThreshold()) {
            graph_.addOutNeighborUnique(neighbour, node);
          }
        }
      }
    }
  }
}

std::unique_ptr<NodeList> Vamana::search(NodeId queryNode, uint64_t k) {
  const HDVector & queryVector = *dataSet_->getRecordViewByIndex(queryNode).vector;
  SearchResults searchResult = greedySearch(queryVector, k);
  return std::make_unique<NodeList>(searchResult.approximateNN);
}

void Vamana::save(std::filesystem::path path) const {
  graph_.save(path);
}

// BIG TODOS
// 1. CUSTOM DATASTRUCTURE
// 2. PARALELLIZE THIS ALGORITHM
