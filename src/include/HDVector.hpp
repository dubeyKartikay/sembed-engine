#ifndef HDVEC
#define HDVEC

#include <cstdint>
#include <vector>

#include "Vector.hpp"

class HDVector : public Vector {
public:
  HDVector(int64_t dimensions);
  HDVector(const std::vector<float> &vec);

  float *getDataPointer() override;
  const float *getDataPointer() const override;
  uint64_t getDimension() const override { return dimensions_; }

  float &operator[](int64_t index) override;
  const float &operator[](int64_t index) const override;

private:
  std::vector<float> data_;
  uint64_t dimensions_ = 0;
};

#endif  // HDVEC
