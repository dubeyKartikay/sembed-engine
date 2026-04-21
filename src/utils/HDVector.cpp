#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "HDVector.hpp"

HDVector::HDVector(int64_t dimensions) {
  if (dimensions < 0) {
    throw std::invalid_argument("vector dimensions must be non-negative");
  }
  data_ = std::vector<float>(static_cast<size_t>(dimensions), 0.0f);
  dimensions_ = static_cast<uint64_t>(dimensions);
}

HDVector::HDVector(const std::vector<float> &vec) {
  data_ = vec;
  dimensions_ = static_cast<uint64_t>(vec.size());
}

float *HDVector::getDataPointer() { return data_.data(); }

float &HDVector::operator[](int64_t index) {
  return data_.at(static_cast<size_t>(index));
}

const float & HDVector::operator[](int64_t index) const {
  return data_.at(static_cast<size_t>(index));
}

float HDVector::distance(const HDVector &left, const HDVector &right) {
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
