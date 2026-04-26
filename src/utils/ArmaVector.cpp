#include "ArmaVector.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

ArmaVector::ArmaVector(int64_t dimensions) {
  if (dimensions < 0) {
    throw std::invalid_argument("vector dimensions must be non-negative");
  }
  m_data.zeros(static_cast<arma::uword>(dimensions));
}

ArmaVector::ArmaVector(const std::vector<float> &values) {
  m_data = arma::fvec(values);
}

float *ArmaVector::data() { return m_data.memptr(); }
const float *ArmaVector::data() const { return m_data.memptr(); }

float &ArmaVector::operator[](int64_t index) {
  if (index < 0 || static_cast<uint64_t>(index) >= dimensions()) {
    throw std::out_of_range("vector index is outside bounds");
  }
  return m_data[static_cast<arma::uword>(index)];
}

const float &ArmaVector::operator[](int64_t index) const {
  if (index < 0 || static_cast<uint64_t>(index) >= dimensions()) {
    throw std::out_of_range("vector index is outside bounds");
  }
  return m_data[static_cast<arma::uword>(index)];
}
