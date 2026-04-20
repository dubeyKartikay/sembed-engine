#include "vamana.hpp"
#include <algorithm>
#include <bitset>
#include <filesystem>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>
Vamana::Vamana(std::unique_ptr<DataSet> datset, int degreeThreshold,
               float distanceThreshold)
    : m_graph(datset->getN(), degreeThreshold) {
  m_dataSet = std::move(datset);
  m_distanceThreshold = distanceThreshold;
  m_searchListSize = 100;
  buildIndex();
}
Vamana::Vamana(std::unique_ptr<DataSet> dataSet, Graph Graph,
               float distanceThreshold)
    : m_graph(Graph) {
  m_dataSet = std::move(dataSet);
  m_distanceThreshold = distanceThreshold;
  m_searchListSize = 100;
  buildIndex();
}
Vamana::Vamana(std::unique_ptr<DataSet> dataSet, std::filesystem::path path,
               float distanceThreshold)
    : m_graph(path) {
  m_dataSet = std::move(dataSet);
  m_distanceThreshold = distanceThreshold;
  m_searchListSize = 100;
  buildIndex();
}
struct customLess {
    HDVector *qNode;
    Vamana *vamana;
    customLess(HDVector *qNode, Vamana *vamana)
        : qNode(qNode), vamana(vamana){};
    bool operator()(int l, int r) {
      const float leftDistance =
          HDVector::distance(*qNode,
                             *(vamana->m_dataSet->getRecordViewByIndex(l).vector));
      const float rightDistance =
          HDVector::distance(*qNode,
                             *(vamana->m_dataSet->getRecordViewByIndex(r).vector));
      if (leftDistance == rightDistance) {
        return l < r;
      }
      return leftDistance < rightDistance;
    }
  };
void Vamana::insertIntoSet(const std::vector<int> &from, std::vector<int> &to,
                           HDVector &comparison_vec) {
  for (const int outNode : from) {

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

SearchResults Vamana::greedySearch(HDVector &node, int k) {
  SearchResults searchResult;
  searchResult.approximateNN.push_back(m_graph.getMediod());
  std::unordered_set<int> visited;
  int maxIter = 0;
  while (maxIter < 10000) {
    int i = 0;
    while (i < m_searchListSize && i < searchResult.approximateNN.size() &&
           visited.count(searchResult.approximateNN[i]) != 0) {
      i++;
    }
    if (i >= m_searchListSize || i >= searchResult.approximateNN.size()) {
      break;
    }

    int node_p_star = searchResult.approximateNN[i];
    visited.insert(node_p_star);
    searchResult.visited.push_back(node_p_star);
    insertIntoSet(m_graph.getOutNeighbours(node_p_star),
                  searchResult.approximateNN, node);
    while (searchResult.approximateNN.size() > m_searchListSize) {
      searchResult.approximateNN.pop_back();
    }
    maxIter++;
  }
  while (searchResult.approximateNN.size() > k) {
    searchResult.approximateNN.pop_back();
  }

  return searchResult;
}

bool Vamana::isToBePruned(int p_dash, int p_star, int p) {
  std::shared_ptr<HDVector> p_dash_vec = m_dataSet->getRecordViewByIndex(p_dash).vector;
  std::shared_ptr<HDVector> p_star_vec = m_dataSet->getRecordViewByIndex(p_star).vector;
  std::shared_ptr<HDVector> p_vec = m_dataSet->getRecordViewByIndex(p).vector;
  float distanceFromP_starToP_dash =
      HDVector::distance(*p_star_vec, *p_dash_vec);
  float distanceFromPToP_dash = HDVector::distance(*p_vec, *p_dash_vec);
  return m_distanceThreshold * distanceFromP_starToP_dash <=
         distanceFromPToP_dash;
}

void Vamana::prune(int node, std::vector<int> &candidateSet) {

  std::shared_ptr<HDVector> p_vec = m_dataSet->getRecordViewByIndex(node).vector;
  const std::vector<int> &outNeighboursP = m_graph.getOutNeighbours(node);
  std::vector<int> candidates = candidateSet;
  insertIntoSet(outNeighboursP, candidates, *p_vec);
  candidates.erase(std::remove(candidates.begin(), candidates.end(), node),
                   candidates.end());
  std::sort(candidates.begin(), candidates.end(),
            customLess(&*p_vec, this));
  m_graph.clearOutNeighbours(node);
  std::unordered_set<int> inOutNeighbours;
  while (candidates.size() > 0) {
    int p_star = candidates[0];
    if (inOutNeighbours.count(p_star) == 0) {
      m_graph.addOutNeighbourUnique(node, p_star);
      inOutNeighbours.insert(p_star);
    }
    if (m_graph.getOutNeighbours(node).size() == m_graph.getDegreeThreshold()) {
      break;
    }

    auto it = candidates.begin();
    while (it != candidates.end()) {
      int p_dash = *it;
      if (isToBePruned(p_dash, p_star, node)) {
        it = candidates.erase(it);

      } else {
        it++;
      }
    }
  }
  if (&candidateSet != &m_graph.getOutNeighbours(node)) {
    candidateSet = candidates;
  }
}

void Vamana::buildIndex() {
  std::vector<int> sigma = getPermutation(m_dataSet->getN());
  for (int &node : sigma) {
    SearchResults greedySearchResult =
        greedySearch(*m_dataSet->getRecordViewByIndex(node).vector, 1);
    prune(node, greedySearchResult.visited);
    for (int neighbour : m_graph.getOutNeighbours(node)) {
      m_graph.addOutNeighbourUnique(neighbour, node);
      if (m_graph.getOutNeighbours(neighbour).size() >
          m_graph.getDegreeThreshold()) {
        prune(neighbour, m_graph.getOutNeighbours(neighbour));
      }
    }
  }
}

// BIG TODOS
// 1. CUSTOM DATASTRUCTURE
// 2. PARALELLIZE THIS ALGORITHM
