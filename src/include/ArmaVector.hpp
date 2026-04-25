#ifndef ARMAVEC
#define ARMAVEC

#include <cstdint>
#include <vector>

#include <armadillo>

#include "vector_view.hpp"

class ArmaVector {
public:
  explicit ArmaVector(int64_t dimensions);
  explicit ArmaVector(const std::vector<float> &values);

  float *data();
  const float *data() const;
  uint64_t dimensions() const { return static_cast<uint64_t>(m_data.n_elem); }
  FloatVectorView view() const {
    return FloatVectorView(m_data.memptr(), dimensions());
  }

  float &operator[](int64_t index);
  const float &operator[](int64_t index) const;

private:
  arma::fvec m_data;
};

#endif  // ARMAVEC
