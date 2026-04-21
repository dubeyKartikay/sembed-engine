#ifndef SEARCH_RESULTS
#define SEARCH_RESULTS

#include "node_types.hpp"

struct SearchResults {
  NodeList approximateNN;
  NodeList visited;
};

#endif  // SEARCH_RESULTS
