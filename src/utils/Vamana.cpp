#include "vamana.hpp"
#include "searchresults.hpp"
#include "utils.hpp"
#include <algorithm>
#include <filesystem>
#include <memory>
#include <unordered_set>
#include <vector>
#include <boost/dynamic_bitset.hpp>

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

struct ScoredNode {
  NodeId node = 0;
  float distanceSquared = 0.0f;
};

bool scoredNodeLess(const ScoredNode &left, const ScoredNode &right) {
  if (left.distanceSquared == right.distanceSquared) {
    return left.node < right.node;
  }
  return left.distanceSquared < right.distanceSquared;
}

void sortNodeListByDistance(const DataSet &dataSet, NodeList &nodes,
                            FloatVectorView comparisonVector) {
  std::vector<ScoredNode> scored;
  scored.reserve(nodes.size());
  std::unordered_set<NodeId> seen;
  for (NodeId node : nodes) {
    if (!seen.insert(node).second) {
      continue;
    }
    scored.push_back(
        {node, squaredDistance(comparisonVector,
                               dataSet.getRecordViewByIndex(node).values)});
  }

  std::sort(scored.begin(), scored.end(), scoredNodeLess);
  nodes.clear();
  nodes.reserve(scored.size());
  for (const ScoredNode &node : scored) {
    nodes.push_back(node.node);
  }
}
}  // namespace

Vamana::Vamana(std::unique_ptr<DataSet> dataSet, uint64_t degreeThreshold,
               float distanceThreshold)
    : m_graph([&]() -> Graph {
        return Graph(checkedDatasetNodeCount(dataSet), degreeThreshold);
      }()) {
  validateDataset(dataSet);
  m_dataSet = std::move(dataSet);
  m_distanceThreshold = distanceThreshold;
  m_searchListSize = 100;
  buildIndex();
}

Vamana::Vamana(std::unique_ptr<DataSet> dataSet, Graph graph,
               float distanceThreshold)
    : m_graph(std::move(graph)) {
  validateDataset(dataSet);
  m_dataSet = std::move(dataSet);
  m_distanceThreshold = distanceThreshold;
  m_searchListSize = 100;
}

Vamana::Vamana(std::unique_ptr<DataSet> dataSet, std::filesystem::path path,
               float distanceThreshold)
    : m_graph(path) {
  validateDataset(dataSet);
  m_dataSet = std::move(dataSet);
  m_distanceThreshold = distanceThreshold;
  m_searchListSize = 100;
}

void Vamana::insertIntoSet(const NodeList &from, SortedBoundedVector &to,
                           FloatVectorView comparisonVector, boost::dynamic_bitset<> &visited) {
  if (from.empty()) {
    return;
  }
  std::vector<NodeId> toInsert;
  std::vector<float> distances;
  toInsert.reserve(from.size());
  for (const NodeId &node : from) {
    if (!visited.test((size_t)node)) {
      toInsert.push_back(node);
      distances.push_back(squaredDistance(comparisonVector,
                                         m_dataSet->getRecordViewByIndex(node).values));
      visited.set((size_t)node);
    }
  }

  for (size_t i = 0; i < toInsert.size(); i++) {
    to.add({distances[i], toInsert[i]});
  }

}

SearchResults Vamana::greedySearch(FloatVectorView query, uint64_t k) {
  SearchResults searchResult(m_searchListSize);
  const OptionalNodeId medoid = m_graph.getMedoid();
  if (!medoid || m_searchListSize == 0 || k == 0) {
    return searchResult;
  }
  searchResult.approximateNN.add({
    squaredDistance(query, m_dataSet->getRecordViewByIndex(*medoid).values),
    *medoid
  });
  boost::dynamic_bitset<> visited(m_graph.getNodeCount());
  while (1) {
    auto nodePStarIndex = searchResult.approximateNN.closestUnexpanded();
    if(nodePStarIndex >= m_searchListSize){
      break;
    }
    auto nodePStar = searchResult.approximateNN[nodePStarIndex];
    visited.set(nodePStar.node);
    searchResult.visited.push_back(nodePStar.node);
    insertIntoSet(m_graph.getOutNeighbors(nodePStar.node),
                  searchResult.approximateNN, query,visited);
  }

  searchResult.approximateNN.trim(k);
  return searchResult;
}

bool Vamana::isToBePruned(NodeId pDash, NodeId pStar, NodeId p) {
  const FloatVectorView pDashValues =
      m_dataSet->getRecordViewByIndex(pDash).values;
  const FloatVectorView pStarValues =
      m_dataSet->getRecordViewByIndex(pStar).values;
  const FloatVectorView pValues = m_dataSet->getRecordViewByIndex(p).values;
  const float alphaSquared = m_distanceThreshold * m_distanceThreshold;
  const float pStarToPDash = squaredDistance(pStarValues, pDashValues);
  const float pToPDash = squaredDistance(pValues, pDashValues);
  return alphaSquared * pStarToPDash <= pToPDash;
}

void Vamana::prune(NodeId node, NodeList &candidateSet) {
  const FloatVectorView pValues = m_dataSet->getRecordViewByIndex(node).values;
  const NodeList &outNeighbors = m_graph.getOutNeighbors(node);
  NodeList candidates = candidateSet;
  insertIntoSet(outNeighbors, candidates, pValues);
  candidates.erase(std::remove(candidates.begin(), candidates.end(), node),
                   candidates.end());
  sortNodeListByDistance(*m_dataSet, candidates, pValues);
  m_graph.clearOutNeighbors(node);
  if (m_graph.getDegreeThreshold() == 0) {
    candidateSet.clear();
    return;
  }
  std::unordered_set<NodeId> inOutNeighbours;
  while (!candidates.empty()) {
    const NodeId pStar = candidates[0];
    if (inOutNeighbours.count(pStar) == 0) {
      m_graph.addOutNeighborUnique(node, pStar);
      inOutNeighbours.insert(pStar);
    }
    if (m_graph.getOutNeighbors(node).size() == m_graph.getDegreeThreshold()) {
      break;
    }

    auto it = candidates.begin();
    while (it != candidates.end()) {
      const NodeId pDash = *it;
      if (isToBePruned(pDash, pStar, node)) {
        it = candidates.erase(it);
      } else {
        ++it;
      }
    }
  }
  if (&candidateSet != &m_graph.mutableOutNeighbors(node)) {
    candidateSet = candidates;
  }
}

void Vamana::buildIndex() {
  auto rng = makeDeterministicRng(
      0x76616d616e61524eULL,
      {m_dataSet->getN(), m_graph.getDegreeThreshold()},
      {m_distanceThreshold});
  NodeList sigma = getPermutation(static_cast<int64_t>(m_dataSet->getN()), rng);
  for (NodeId &node : sigma) {
    SearchResults greedySearchResult =
        greedySearch(m_dataSet->getRecordViewByIndex(node).values, 1);
    prune(node, greedySearchResult.visited);
    for (NodeId neighbour : m_graph.getOutNeighbors(node)) {
      try {
        m_graph.addOutNeighborUnique(neighbour, node);
        //wtf
      } catch (const std::invalid_argument &) {
        prune(neighbour, m_graph.mutableOutNeighbors(neighbour));
        if (std::find(m_graph.getOutNeighbors(neighbour).begin(),
                      m_graph.getOutNeighbors(neighbour).end(),
                      node) == m_graph.getOutNeighbors(neighbour).end()) {
          if (m_graph.getOutNeighbors(neighbour).size() <
              m_graph.getDegreeThreshold()) {
            m_graph.addOutNeighborUnique(neighbour, node);
          }
        }
      }
    }
  }
}

std::unique_ptr<NodeList> Vamana::search(NodeId queryNode, uint64_t k) {
  const FloatVectorView queryVector =
      m_dataSet->getRecordViewByIndex(queryNode).values;
  SearchResults searchResult = greedySearch(queryVector, k);
  return std::make_unique<NodeList>(searchResult.approximateNN);
}

void Vamana::save(std::filesystem::path path) const {
  m_graph.save(path);
}
