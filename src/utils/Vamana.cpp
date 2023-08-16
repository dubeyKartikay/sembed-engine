#include "vamana.hpp"
#include <algorithm>
#include <bitset>
#include <filesystem>
#include <memory>
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
      return HDVector::distance(*qNode,
                                *(vamana->m_dataSet->getHDVecByIndex(l))) <
             HDVector::distance(*qNode,
                                *(vamana->m_dataSet->getHDVecByIndex(r)));
    }
  };
  for (int outNode : NOut) {
    ANNset.insert(std::lower_bound(ANNset.begin(), ANNset.end(), outNode,
                                   customLess(&node, this)),
                  outNode);
  }
}
SearchResults Vamana::greedySearch(HDVector node, int k) {
  SearchResults searchResult;
  searchResult.approximateNN.push_back(m_graph.getMediod());
  std::vector<bool> visited(m_dataSet->getN(), false);
  int i = 0;
  while (i < m_searchListSize) {
    while (visited[searchResult.approximateNN[i]]) {
      i++;
    } // L/V
    int node_p_star = searchResult.approximateNN[i];
    searchResult.visited.push_back(node_p_star);
    insertIntoANN(m_graph.getOutNeighbours(node_p_star),
                  searchResult.approximateNN, node);
    visited[node_p_star] = true;
    if (searchResult.approximateNN.size() > m_searchListSize) {
      // trim searchResult
      while (searchResult.approximateNN.size() > m_searchListSize) {
        searchResult.approximateNN.pop_back();
      }
    }
  }
  // trim searchResult.approximateNN to k,
  while (searchResult.approximateNN.size() > k) {
    searchResult.approximateNN.pop_back();
  }
  return searchResult;
}
