#ifndef HDVEC
#define HDVEC

#include <cstdint>
#include <vector>

#include "vector_view.hpp"

class HDVector {
public:
  explicit HDVector(int64_t dimensions);
  explicit HDVector(const std::vector<float> &values);

  float *data();
  const float *data() const;
  uint64_t dimensions() const { return static_cast<uint64_t>(m_data.size()); }
  FloatVectorView view() const {
    return FloatVectorView(m_data.data(), dimensions());
  }

  float &operator[](int64_t index);
  const float &operator[](int64_t index) const;

private:
  std::vector<float> m_data;
};

#endif  // HDVEC
