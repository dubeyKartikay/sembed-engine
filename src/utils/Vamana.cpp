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
}
Vamana::Vamana(std::unique_ptr<DataSet> dataSet, Graph Graph,
               float distanceThreshold)
    : m_graph(Graph) {
  m_dataSet = std::move(dataSet);
  m_distanceThreshold = distanceThreshold;
}
Vamana::Vamana(std::unique_ptr<DataSet> dataSet, std::filesystem::path path,
               float distanceThreshold)
    : m_graph(path) {
  m_dataSet = std::move(dataSet);
  m_distanceThreshold = distanceThreshold;
}

void Vamana::insertIntoSet(std::vector<int> &from, std::vector<int> &to,
                           HDVector &comparison_vec) {
  struct customLess {
    HDVector *qNode;
    Vamana *vamana;
    customLess(HDVector *qNode, Vamana *vamana)
        : qNode(qNode), vamana(vamana){};
    bool operator()(const int &l, const int &r) {
      float d =
          HDVector::distance(*qNode, *(vamana->m_dataSet->getHDVecByIndex(l))) -
          HDVector::distance(*qNode, *(vamana->m_dataSet->getHDVecByIndex(r)));
      if (d == 0) {
        return l < r;
      } else {
        return d < 0;
      }
    }
  };
  for (int outNode : from) {
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

SearchResults Vamana::greedySearch(HDVector node, int k) {
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
  std::shared_ptr<HDVector> p_dash_vec = m_dataSet->getHDVecByIndex(p_dash);
  std::shared_ptr<HDVector> p_star_vec = m_dataSet->getHDVecByIndex(p_star);
  std::shared_ptr<HDVector> p_vec = m_dataSet->getHDVecByIndex(p);
  float distanceFromP_starToP_dash =
      HDVector::distance(*p_star_vec, *p_dash_vec);
  float distanceFromPToP_dash = HDVector::distance(*p_vec, *p_dash_vec);
  return m_distanceThreshold * distanceFromP_starToP_dash <=
         distanceFromPToP_dash;
}

void Vamana::prune(int node, std::vector<int> &candidateSet) {
  std::shared_ptr<HDVector> p_vec = m_dataSet->getHDVecByIndex(node);
  std::vector<int> OutNeighboursP = m_graph.getOutNeighbours(node);
  insertIntoSet(OutNeighboursP, candidateSet, *p_vec);
  OutNeighboursP.clear();
  std::unordered_set<int> inOutNeighbours;
  while (candidateSet.size() > 0) {
    int p_star = candidateSet[0];
    if (inOutNeighbours.count(p_star) == 0) {
      OutNeighboursP.push_back(p_star);
      inOutNeighbours.insert(p_star);
    }
    if (OutNeighboursP.size() == m_graph.getDegreeThreshold()) {
      break;
    }

    auto it = candidateSet.begin();
    while (it != candidateSet.end()) {
      int p_dash = *it;
      if (isToBePruned(p_dash, p_star, node)) {
        it = candidateSet.erase(it);

      } else {
        it++;
      }
    }
  }
}
