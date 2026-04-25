#ifndef VECTOR_VIEW_HPP
#define VECTOR_VIEW_HPP

#include <cmath>
#include <cstdint>
#include <stdexcept>

class FloatVectorView {
public:
  FloatVectorView() = default;
  FloatVectorView(const float *data, uint64_t dimensions)
      : m_data(data), m_dimensions(dimensions) {}

  const float *data() const { return m_data; }
  uint64_t dimensions() const { return m_dimensions; }
  bool empty() const { return m_dimensions == 0; }

  const float &operator[](uint64_t index) const {
    if (index >= m_dimensions || m_data == nullptr) {
      throw std::out_of_range("vector view index is outside bounds");
    }
    return m_data[index];
  }

private:
  const float *m_data = nullptr;
  uint64_t m_dimensions = 0;
};

inline float squaredDistance(FloatVectorView left, FloatVectorView right) {
  if (left.dimensions() != right.dimensions()) {
    throw std::invalid_argument(
        "vector dimensions do not match, cannot compute distance");
  }

  double distanceSquared = 0.0;
  const float *leftData = left.data();
  const float *rightData = right.data();
  for (uint64_t i = 0; i < left.dimensions(); ++i) {
    const double delta =
        static_cast<double>(leftData[i]) - static_cast<double>(rightData[i]);
    distanceSquared += delta * delta;
  }
  return static_cast<float>(distanceSquared);
}

inline float euclideanDistance(FloatVectorView left, FloatVectorView right) {
  if (left.dimensions() != right.dimensions()) {
    throw std::invalid_argument(
        "vector dimensions do not match, cannot compute distance");
  }

  double distanceSquared = 0.0;
  const float *leftData = left.data();
  const float *rightData = right.data();
  for (uint64_t i = 0; i < left.dimensions(); ++i) {
    const double delta =
        static_cast<double>(leftData[i]) - static_cast<double>(rightData[i]);
    distanceSquared += delta * delta;
  }
  return static_cast<float>(std::sqrt(distanceSquared));
}

#endif  // VECTOR_VIEW_HPP
