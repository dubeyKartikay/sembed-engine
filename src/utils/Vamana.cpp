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

void Vamana::insertIntoANN(std::vector<int> &NOut, std::vector<int> &ANNset,
                           HDVector &node) {
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
  for (int outNode : NOut) {
    auto pos = std::lower_bound(ANNset.begin(), ANNset.end(), outNode,
                                customLess(&node, this));
    if (pos == ANNset.end()) {
      ANNset.push_back(outNode);
    } else {
      if (*pos != outNode) {
        ANNset.insert(pos, outNode);
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
    insertIntoANN(m_graph.getOutNeighbours(node_p_star),
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
