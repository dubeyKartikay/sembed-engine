#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cstdint>

class Vector {
public:
  virtual ~Vector() = default;

  virtual float *getDataPointer() = 0;
  virtual const float *getDataPointer() const = 0;
  virtual uint64_t getDimension() const = 0;

  virtual float &operator[](int64_t index) = 0;
  virtual const float &operator[](int64_t index) const = 0;

  static float distance(const Vector &left, const Vector &right);
};

#endif  // VECTOR_HPP
