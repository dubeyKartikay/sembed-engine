#ifndef SEARCH_RESULTS
#define SEARCH_RESULTS

#include "node_types.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <boost/dynamic_bitset.hpp>

struct Neighbour{
  float distance;
  NodeId node;
  bool marked;
  Neighbour(){
    node = std::numeric_limits<uint64_t>::max();
    distance = std::numeric_limits<float>::max();
    marked = false;
  }
  Neighbour(float distance, NodeId node){
    this->distance = distance;
    this->node = node;
    marked = false;
  }

  bool operator<(const Neighbour &other) const {
    if (distance == other.distance) {
      return node < other.node;
    }
    return distance < other.distance;
  }

  // static std::vector<Neighbour> fromNodeList(const NodeList & nodes, boost::dynamic_bitset<> &blacklist){
  //
  // }
};

class SortedBoundedVector{
  std::vector<Neighbour> neighbours;
  uint64_t capacity;
  uint64_t size;
  uint64_t cursor;
  public:
  SortedBoundedVector(uint64_t capacity){
    this->capacity = capacity;
    this->size = 0;
    neighbours.resize(capacity + 1);
    cursor = std::numeric_limits<uint64_t>::max();
  }

  uint64_t getCursor() const { return cursor; }
  uint64_t getSize() const { return size; }

  bool add(Neighbour neighbour){
    if(size == capacity && neighbour.distance > neighbours[size - 1].distance){
      return false;
    }
    uint64_t l = 0, r = size;
    while (l < r) {
      uint64_t m = l + (r - l) / 2;
      if (neighbours[m] < neighbour) {
        l = m + 1;
      } else if (neighbours[m].node == neighbour.node) {
        return false;
      }
      else {
        r = m;
      }
    }

    if(l>=capacity){
      return false;
    }

    if(l == size && size < capacity){
      neighbours[l] = neighbour;
      size++;
      if (l < cursor) {
        cursor = l;
      }
      return true;
    }

    std::memmove(neighbours.data() + l + 1, neighbours.data() + l,
                 (size - l) * sizeof(Neighbour));
    neighbours[l] = neighbour;
    if (size < capacity) {
      ++size;
    }
    if (l < cursor) {
      cursor = l;
    }
    return true;
  }

  void trim(uint64_t size){
    size = std::min(size, std::min(capacity, this->size));
    this->size = size;
    neighbours.resize(size);
  }
  
  const Neighbour &operator[](uint64_t index) const { return neighbours[index]; }

  uint64_t closestUnexpanded() {
    neighbours[cursor].marked = true;
    uint64_t pre = cursor;
    while (cursor < size && neighbours[cursor].marked) {
      cursor++;
    }
    return pre;
  }


};

struct SearchResults {
  SortedBoundedVector approximateNN;
  std::vector<Neighbour> visited;
  boost::dynamic_bitset<> visitedBitset;
  SearchResults(uint64_t searchListSize): approximateNN(searchListSize){
    visited.reserve(2*searchListSize);
  };
};


#endif  // SEARCH_RESULTS
