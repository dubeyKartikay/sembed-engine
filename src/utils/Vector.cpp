#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "Vector.hpp"

float Vector::distance(const Vector &left, const Vector &right) {
  if (left.getDimension() != right.getDimension()) {
    throw std::invalid_argument(
        "vector dimensions do not match, cannot compute distance");
  }

  double distanceSquared = 0.0;
  for (uint64_t i = 0; i < left.getDimension(); ++i) {
    const double delta =
        static_cast<double>(left[static_cast<int64_t>(i)]) -
        static_cast<double>(right[static_cast<int64_t>(i)]);
    distanceSquared += delta * delta;
  }

  return static_cast<float>(std::sqrt(distanceSquared));
}
