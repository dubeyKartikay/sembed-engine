#ifndef HDVEC
#define HDVEC

#include <cstdint>
#include <vector>

class HDVector {
public:
  HDVector(int64_t dimensions);
  HDVector(const std::vector<float> &vec);

  float *getDataPointer();
  uint64_t getDimension() const { return dimensions_; }

  float &operator[](int64_t index);
  const float &operator[](int64_t index) const;

  static float distance(const HDVector &left, const HDVector &right);

private:
  std::vector<float> data_;
  uint64_t dimensions_ = 0;
};

#endif  // HDVEC
