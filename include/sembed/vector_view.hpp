#ifndef SEMBED_VECTOR_VIEW_HPP
#define SEMBED_VECTOR_VIEW_HPP

#include <cstdint>
#include <vector>

namespace sembed {

class FloatVectorView {
 public:
  FloatVectorView() = default;
  FloatVectorView(const float* data, uint64_t dimensions)
      : data_(data), dimensions_(dimensions) {}
  explicit FloatVectorView(const std::vector<float>& values)
      : data_(values.data()), dimensions_(values.size()) {}

  const float* data() const { return data_; }
  uint64_t dimensions() const { return dimensions_; }
  bool empty() const { return dimensions_ == 0; }

 private:
  const float* data_ = nullptr;
  uint64_t dimensions_ = 0;
};

}  // namespace sembed

#endif  // SEMBED_VECTOR_VIEW_HPP
