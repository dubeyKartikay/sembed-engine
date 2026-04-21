#include <cstdint>
#include <vector>
#include <memory>
#ifndef HDVEC
#define HDVEC

class HDVector{
  private:
    std::vector<float> m_data;
    uint64_t dimentions;
  public:
    float * getDataPointer();
    uint64_t getDimention() const {
      return this->dimentions;
    }
    HDVector(int64_t dimentions);
    HDVector(const std::vector<float> & vec);
    float & operator[](int64_t index);
    const float & operator[](int64_t index) const;
    // HDVector operator+(HDVector & other); 
    // HDVector operator-(HDVector & other);
    static float distance(const HDVector & vec1,const HDVector &vec2);
};

#endif // DEBUG
