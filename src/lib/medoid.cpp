#include "medoid.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "vector_view.hpp"

namespace sembed_internal {

OptionalNodeId computeMedoidCpu(const DataSet& dataSet) {
  const uint64_t n = dataSet.getN();
  if (n == 0) {
    return std::nullopt;
  }
  const uint64_t dims = dataSet.getDimensions();
  if (n == 1 || dims == 0) {
    return NodeId{0};
  }

  std::vector<double> sum(static_cast<size_t>(dims), 0.0);
  const float* raw = dataSet.data();
  if (raw != nullptr) {
    for (uint64_t i = 0; i < n; ++i) {
      const float* row = raw + static_cast<size_t>(i * dims);
      for (uint64_t d = 0; d < dims; ++d) {
        sum[static_cast<size_t>(d)] += static_cast<double>(row[d]);
      }
    }
  } else {
    for (uint64_t i = 0; i < n; ++i) {
      const FloatVectorView values = dataSet.getRecordViewByIndex(i).values;
      const float* row = values.data();
      for (uint64_t d = 0; d < dims; ++d) {
        sum[static_cast<size_t>(d)] += static_cast<double>(row[d]);
      }
    }
  }

  std::vector<float> centroid(static_cast<size_t>(dims), 0.0F);
  const double invN = 1.0 / static_cast<double>(n);
  for (uint64_t d = 0; d < dims; ++d) {
    centroid[static_cast<size_t>(d)] =
        static_cast<float>(sum[static_cast<size_t>(d)] * invN);
  }

  const FloatVectorView centroidView(centroid.data(), dims);
  NodeId bestNode = 0;
  float bestDist = std::numeric_limits<float>::infinity();
  if (raw != nullptr) {
    for (uint64_t i = 0; i < n; ++i) {
      const float dist = squaredDistance(
          centroidView,
          FloatVectorView(raw + static_cast<size_t>(i * dims), dims));
      if (dist < bestDist) {
        bestDist = dist;
        bestNode = i;
      }
    }
  } else {
    for (uint64_t i = 0; i < n; ++i) {
      const float dist =
          squaredDistance(centroidView, dataSet.getRecordViewByIndex(i).values);
      if (dist < bestDist) {
        bestDist = dist;
        bestNode = i;
      }
    }
  }
  return bestNode;
}

OptionalNodeId computeMedoid(const DataSet& dataSet) {
#ifdef SEMBED_ENABLE_METAL
  return computeMedoidMetal(dataSet);
#else
  return computeMedoidCpu(dataSet);
#endif
}

}  // namespace sembed_internal
