#include "dataset.hpp"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "utils.hpp"
InMemoryDataSet::InMemoryDataSet(fs::path path){

        if(!isValidPath(path)){
            throw std::invalid_argument("The path does not exist");
        }
        m_file.open(path);
        if(!m_file.is_open()){
            throw std::runtime_error("could not open the file provided");
        }
        m_file.read((char*)&this->n,sizeof(this->n));
        m_file.read((char*)&this->dimentions,sizeof(this->dimentions));
        m_data = readDataFromFile();
        m_file.close();

}
std::vector<HDVector*> * InMemoryDataSet::readDataFromFile(){
  std::vector<HDVector * > * vec = new std::vector<HDVector *>(this->getN(),new HDVector(this->getDimentions()));
  float * buf = (float *)malloc(this->getDimentions() * this->getN() * sizeof(float));
  if(buf == NULL){
    throw std::runtime_error("Cannot allocated the required memoory to load the dataset into memory");
  }
  m_file.read(reinterpret_cast<char *>(buf), this->getDimentions() * this->getN() * sizeof(float));
  
  for(int i = 0;i < this->getN(); i++){
    std::copy_n(buf + i*this->getDimentions(),this->getDimentions(),vec->at(i)->getDataPointer());
  }
  return vec;
  
}
const HDVector& InMemoryDataSet::getHDVecByIndex(const int & index) const {
  HDVector * vec = this->m_data->at(index);
  auto &ret = *vec;
  return ret;
}

 const std::vector<HDVector*> & InMemoryDataSet::getNHDVectorsFromIndex(const int  &index,const int & n) const  {
        std::vector<HDVector*> * vec = new std::vector<HDVector*>( this->m_data->begin() + index , this->m_data->begin() +n );
        auto & ret = *vec;
        return ret;
 }
