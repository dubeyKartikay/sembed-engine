#include "vamana.hpp"
#include "searchresults.hpp"
#include "utils.hpp"
#include "vector_view.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
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
  // boost::dynamic_bitset<> visited(m_graph.getNodeCount());
  searchResult.visitedBitset = boost::dynamic_bitset<>(m_graph.getNodeCount());
  while (1) {
    auto nodePStarIndex = searchResult.approximateNN.closestUnexpanded();
    if(nodePStarIndex >= searchResult.approximateNN.getSize()){
      break;
    }
    auto nodePStar = searchResult.approximateNN[nodePStarIndex];
    searchResult.visitedBitset.set(nodePStar.node);
    searchResult.visited.push_back(nodePStar);
    insertIntoSet(m_graph.getOutNeighbors(nodePStar.node),
                  searchResult.approximateNN, query,searchResult.visitedBitset);
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

void Vamana::prune(NodeId node,const std::vector<Neighbour> &candidates) {
  m_graph.clearOutNeighbors(node);
  boost::dynamic_bitset<> visited(m_graph.getNodeCount());
  boost::dynamic_bitset<> deletedCandidates(candidates.size());
  uint64_t next = 0;
  while (next < candidates.size()) {
    const Neighbour pStar = candidates[next];
    if (!visited.test(pStar.node)) {
      m_graph.addOutNeighborUnique(node, pStar.node);
      visited.set(pStar.node);
    }
    if (m_graph.getOutNeighbors(node).size() == m_graph.getDegreeThreshold()) {
      break;
    }

    for (uint64_t i = 0; i < candidates.size(); i++){
      const Neighbour pDash = candidates[i];
      if (isToBePruned(pDash.node, pStar.node, node)) {
        deletedCandidates.set(i);
      }
    }

    while (next < candidates.size() && deletedCandidates.test(next)) {
      next++;
    }
  }
}

void Vamana::buildIndex() {
  auto rng = makeDeterministicRng(
      0x76616d616e61524eULL, {m_dataSet->getN(), m_graph.getDegreeThreshold()},
      {m_distanceThreshold});
  NodeList sigma = getPermutation(static_cast<int64_t>(m_dataSet->getN()), rng);
  for (NodeId &node : sigma) {
    FloatVectorView nodeView = m_dataSet->getRecordViewByIndex(node).values;
    SearchResults greedySearchResult = greedySearch(nodeView, 1);

    std::vector<Neighbour> candidates;
    candidates.reserve(m_graph.getDegreeThreshold() +
                       greedySearchResult.visited.size());
    for (const auto &neighbour : m_graph.getOutNeighbors(node)) {
      candidates.emplace_back(
          squaredDistance(nodeView,
                          m_dataSet->getRecordViewByIndex(neighbour).values),
          neighbour);
    }
    candidates.insert(candidates.end(), greedySearchResult.visited.begin(),
                      greedySearchResult.visited.end());
    std::sort(candidates.begin(), candidates.end());
    prune(node, candidates);
    for (NodeId neighbour : m_graph.getOutNeighbors(node)) {
      m_graph.addOutNeighborUnique(neighbour, node);
      if (m_graph.getOutNeighbors(neighbour).size() <
          m_graph.getDegreeThreshold()) {
        continue;
      }

      std::vector<Neighbour> candidates;
      candidates.reserve(m_graph.getDegreeThreshold() + 1);
      FloatVectorView neighbourView =
          m_dataSet->getRecordViewByIndex(neighbour).values;
      for (const auto &n : m_graph.getOutNeighbors(neighbour)) {
        candidates.emplace_back(
            squaredDistance(neighbourView,
                            m_dataSet->getRecordViewByIndex(n).values),
            n);
      }
      prune(neighbour, candidates);
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

// std::unique_ptr<NodeList> Vamana::search(NodeId queryNode, uint64_t k) {
//   const FloatVectorView queryVector =
//       m_dataSet->getRecordViewByIndex(queryNode).values;
//   SearchResults searchResult = greedySearch(queryVector, k);
//   std::vector<NodeList> results;
//   results.reserve(searchResult.approximateNN.getSize());
//   for (uint64_t i = 0; i < searchResult.approximateNN.getSize(); i++) {
//     results.emplace_back(searchResult.approximateNN[i].node);
//   }
//
//
//   return std::make_unique<NodeList>(results);
// }

void Vamana::save(std::filesystem::path path) const {
  m_graph.save(path);
}
