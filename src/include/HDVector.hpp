#include <vector>
#ifndef HDVEC
#define HDVEC
class HDVector{
  private:
    std::vector<float> * m_data;
    int dimentions;
  public:
    float * getDataPointer();
    const int & getDimention() const {
      return this->dimentions;
    }
    HDVector(const int & dimentions);
    HDVector(const std::vector<float> & vec);
    float & operator[](int index);
    const float & operator[](int index) const;
    // HDVector operator+(HDVector & other); 
    // HDVector operator-(HDVector & other);
    static float distance(const HDVector & vec1,const HDVector &vec2);
};

#endif // DEBUG
