#ifndef SEMBED_DATASET_VIEW_HPP
#define SEMBED_DATASET_VIEW_HPP

#include <cstdint>

namespace sembed {

struct DatasetView {
  const int64_t* recordIds = nullptr;
  const float* vectors = nullptr;
  uint64_t size = 0;
  uint64_t dimensions = 0;
};

}  // namespace sembed

#endif  // SEMBED_DATASET_VIEW_HPP
