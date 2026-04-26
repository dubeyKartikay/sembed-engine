#include "HDVector.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

HDVector::HDVector(int64_t dimensions) {
  if (dimensions < 0) {
    throw std::invalid_argument("vector dimensions must be non-negative");
  }
  m_data = std::vector<float>(static_cast<size_t>(dimensions), 0.0f);
}

HDVector::HDVector(const std::vector<float> &values) { m_data = values; }

float *HDVector::data() { return m_data.data(); }
const float *HDVector::data() const { return m_data.data(); }

float &HDVector::operator[](int64_t index) {
  if (index < 0) {
    throw std::out_of_range("vector index is outside bounds");
  }
  return m_data.at(static_cast<size_t>(index));
}

const float &HDVector::operator[](int64_t index) const {
  if (index < 0) {
    throw std::out_of_range("vector index is outside bounds");
  }
  return m_data.at(static_cast<size_t>(index));
}
