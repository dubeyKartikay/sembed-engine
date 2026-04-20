#include "HDVector.hpp"
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
HDVector::HDVector(int64_t dimentions){
  if (dimentions < 0) {
    throw std::invalid_argument("vector dimensions must be non-negative");
  }
  this->m_data = std::vector<float>(static_cast<size_t>(dimentions),0);
  this->dimentions = static_cast<uint64_t>(dimentions);
}
HDVector::HDVector(const std::vector<float> & vec ){
  this->m_data = vec;
  this->dimentions = static_cast<uint64_t>(vec.size());
  
}

float * HDVector::getDataPointer(){
  return this->m_data.data();
}

float & HDVector::operator[](int64_t index){
  return this->m_data.at(static_cast<size_t>(index));
}
const float & HDVector::operator[](int64_t index) const {
  return this->m_data.at(static_cast<size_t>(index));
}
float HDVector::distance(const HDVector &vec1, const HDVector &vec2){
  if(vec1.getDimention() != vec2.getDimention()){
    throw std::invalid_argument("Vector dimentions do not match, can't find distace between vectors of different dimentions");
  }
  double dist = 0;
  for (uint64_t i = 0; i < vec1.getDimention(); ++i) {
    const double delta =
        static_cast<double>(vec1[static_cast<int64_t>(i)]) -
        static_cast<double>(vec2[static_cast<int64_t>(i)]);
    dist += delta * delta;
  }
  dist = std::sqrt(dist);
  return dist;
}
