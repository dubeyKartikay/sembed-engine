#ifndef SEMBED_QUERY_HPP
#define SEMBED_QUERY_HPP

#include <cstdint>
#include <vector>

#include <sembed/index.hpp>
#include <sembed/vector_view.hpp>

namespace sembed {

struct QueryConfig {
  uint64_t k = 10;
  // Zero reuses the search-list size stored in the index.
  uint64_t searchListSize = 0;
};

struct Neighbor {
  uint64_t node = 0;
  int64_t recordId = 0;
  float distance = 0.0F;
};

std::vector<Neighbor> query(const Index& index, FloatVectorView queryVector,
                            const QueryConfig& config = {});

}  // namespace sembed

#endif  // SEMBED_QUERY_HPP
