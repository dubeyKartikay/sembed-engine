#include "vamana.hpp"
#include "searchresults.hpp"
#include "utils.hpp"
#include "vector_view.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
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

void sortAndDeduplicateCandidates(std::vector<Neighbour> &candidates) {
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end(),
                               [](const Neighbour &left,
                                  const Neighbour &right) {
                                 return left.node == right.node;
                               }),
                   candidates.end());
}

constexpr std::array<char, 8> kIndexMagic = {'S', 'E', 'M', 'B',
                                              'E', 'D', '0', '1'};
constexpr uint32_t kIndexVersion = 1;

}  // namespace

Vamana::Vamana(std::unique_ptr<DataSet> dataSet, uint64_t degreeThreshold,
               float distanceThreshold, uint64_t searchListSize)
    : m_graph([&]() -> Graph {
        return Graph(checkedDatasetNodeCount(dataSet), degreeThreshold);
      }()) {
  validateDataset(dataSet);
  m_dataSet = std::move(dataSet);
  m_distanceThreshold = distanceThreshold;
  m_searchListSize = searchListSize;
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

Vamana::Vamana(std::filesystem::path path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("could not open the index file provided");
  }

  std::array<char, kIndexMagic.size()> magic{};
  uint32_t version = 0;
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  input.read(reinterpret_cast<char *>(&version), sizeof(version));
  input.read(reinterpret_cast<char *>(&m_distanceThreshold),
             sizeof(m_distanceThreshold));
  input.read(reinterpret_cast<char *>(&m_searchListSize),
             sizeof(m_searchListSize));
  if (!input) {
    throw std::runtime_error("failed to read index header");
  }
  if (magic != kIndexMagic) {
    throw std::runtime_error("index file has invalid magic");
  }
  if (version != kIndexVersion) {
    throw std::runtime_error("index file version is not supported");
  }
  if (!(m_distanceThreshold > 0.0F) ||
      !std::isfinite(m_distanceThreshold)) {
    throw std::runtime_error("index file has invalid distance threshold");
  }
  if (m_searchListSize == 0) {
    throw std::runtime_error("index file has invalid search list size");
  }

  m_dataSet = std::make_unique<FlatDataSet>(input);
  m_graph = Graph(input);
  if (m_dataSet->getN() != m_graph.getNodeCount()) {
    throw std::runtime_error("index graph and dataset sizes do not match");
  }

  char trailing = '\0';
  if (input.read(&trailing, 1)) {
    throw std::runtime_error("index file contains trailing data");
  }
  if (!input.eof()) {
    throw std::runtime_error("failed to validate index file size");
  }
}

void Vamana::insertIntoSet(const NodeList &from, SortedBoundedVector &to,
                           FloatVectorView comparisonVector,
                           std::vector<bool> &visited) const {
  if (from.empty()) {
    return;
  }
  std::vector<NodeId> toInsert;
  std::vector<float> distances;
  toInsert.reserve(from.size());
  for (const NodeId &node : from) {
    if (!visited.at(static_cast<size_t>(node))) {
      toInsert.push_back(node);
      distances.push_back(squaredDistance(comparisonVector,
                                         m_dataSet->getRecordViewByIndex(node).values));
      visited[static_cast<size_t>(node)] = true;
    }
  }

  for (size_t i = 0; i < toInsert.size(); i++) {
    to.add({distances[i], toInsert[i]});
  }

}

SearchResults Vamana::greedySearch(FloatVectorView query, uint64_t k) const {
  return greedySearch(query, k, m_searchListSize);
}

SearchResults Vamana::greedySearch(FloatVectorView query, uint64_t k,
                                   uint64_t searchListSize) const {
  if (query.dimensions() != m_dataSet->getDimensions()) {
    throw std::invalid_argument("query dimensions must match the index");
  }
  SearchResults searchResult(searchListSize);
  const OptionalNodeId medoid = m_graph.getMedoid();
  if (!medoid || searchListSize == 0 || k == 0) {
    return searchResult;
  }
  searchResult.approximateNN.add({
    squaredDistance(query, m_dataSet->getRecordViewByIndex(*medoid).values),
    *medoid
  });
  searchResult.visitedBitset.assign(
      static_cast<size_t>(m_graph.getNodeCount()), false);
  while (1) {
    auto nodePStarIndex = searchResult.approximateNN.closestUnexpanded();
    if(nodePStarIndex >= searchResult.approximateNN.getSize()){
      break;
    }
    auto nodePStar = searchResult.approximateNN[nodePStarIndex];
    searchResult.visitedBitset[static_cast<size_t>(nodePStar.node)] = true;
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
  const uint64_t degreeThreshold = m_graph.getDegreeThreshold();
  std::vector<bool> deletedCandidates(candidates.size(), false);
  uint64_t next = 0;
  uint64_t selected = 0;
  while (next < candidates.size()) {
    while (next < candidates.size() && deletedCandidates[next]) {
      next++;
    }
    if (next == candidates.size()) {
      break;
    }

    const Neighbour pStar = candidates[next];
    m_graph.addOutNeighborUnique(node, pStar.node);
    deletedCandidates[next] = true;
    ++selected;
    if (selected == degreeThreshold) {
      break;
    }

    for (uint64_t i = next + 1; i < candidates.size(); i++){
      if (deletedCandidates[i]) {
        continue;
      }
      const Neighbour pDash = candidates[i];
      if (isToBePruned(pDash.node, pStar.node, node)) {
        deletedCandidates[i] = true;
      }
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
    for (const Neighbour &neighbour : greedySearchResult.visited) {
      if (neighbour.node == node) {
        continue;
      }
      candidates.push_back(neighbour);
    }
    sortAndDeduplicateCandidates(candidates);
    prune(node, candidates);
    for (NodeId neighbour : m_graph.getOutNeighbors(node)) {
      m_graph.addOutNeighborUnique(neighbour, node);
      if (m_graph.getOutNeighbors(neighbour).size() <=
          m_graph.getDegreeThreshold()) {
        continue;
      }

      const NodeList &neighbourOutNeighbors = m_graph.getOutNeighbors(neighbour);
      std::vector<Neighbour> candidates;
      candidates.reserve(neighbourOutNeighbors.size());
      FloatVectorView neighbourView =
          m_dataSet->getRecordViewByIndex(neighbour).values;
      for (const auto &n : neighbourOutNeighbors) {
        if (n == neighbour) {
          continue;
        }
        candidates.emplace_back(
            squaredDistance(neighbourView,
                            m_dataSet->getRecordViewByIndex(n).values),
            n);
      }
      sortAndDeduplicateCandidates(candidates);
      prune(neighbour, candidates);
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

void Vamana::saveIndex(std::filesystem::path path) const {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("could not open the index file for writing");
  }

  output.write(kIndexMagic.data(),
               static_cast<std::streamsize>(kIndexMagic.size()));
  output.write(reinterpret_cast<const char *>(&kIndexVersion),
               sizeof(kIndexVersion));
  output.write(reinterpret_cast<const char *>(&m_distanceThreshold),
               sizeof(m_distanceThreshold));
  output.write(reinterpret_cast<const char *>(&m_searchListSize),
               sizeof(m_searchListSize));
  if (!output) {
    throw std::runtime_error("failed to write index header");
  }

  m_dataSet->save(output);
  m_graph.save(output);
  output.flush();
  if (!output) {
    throw std::runtime_error("failed to write index file");
  }
}
