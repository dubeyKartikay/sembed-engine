#include "HDVector.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>
HDVector::HDVector(const int & dimentions){
  this->m_data = std::vector<float>(dimentions,0);
  this->dimentions = dimentions;
}
HDVector::HDVector(const std::vector<float> & vec ){
  this->m_data = vec;
  this->dimentions = vec.size();
  
}

float * HDVector::getDataPointer(){
  return this->m_data.data();
}

float & HDVector::operator[](int index){
  return this->m_data.at(index);
}
const float & HDVector::operator[](int index) const {
  return this->m_data.at(index);
}
float HDVector::distance(const HDVector &vec1, const HDVector &vec2){
  if(vec1.getDimention() != vec2.getDimention()){
    throw std::invalid_argument("Vector dimentions do not match, can't find distace between vectors of different dimentions");
  }
  double dist = 0;
  for (int i = 1; i < vec1.getDimention(); i++) {
    dist += (vec1[i]-vec2[i])*(vec1[i]-vec2[i]);
  }
  dist = std::sqrt(dist);
  return dist;
}
