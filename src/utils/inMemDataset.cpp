#include "dataset.hpp"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
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
        readDataFromFile(); // reads data to m_data
        m_file.close();

}
void InMemoryDataSet::readDataFromFile(){
  float * buf = (float *)malloc(this->getDimentions() * this->getN() * sizeof(float));
  if(buf == NULL){
    throw std::runtime_error("Cannot allocated the required memoory to load the dataset into memory");
  }
  m_file.read(reinterpret_cast<char *>(buf), this->getDimentions() * this->getN() * sizeof(float));
  std::cout << "BUF 0" << buf[0] << std::endl; 
  for(int i = 0;i < this->getN(); i++){
    std::shared_ptr<HDVector> hdv = std::make_shared<HDVector>(dimentions);
    std::copy_n(buf + i*this->getDimentions(),this->getDimentions(),hdv->getDataPointer());
    m_data.push_back(hdv);
  }
  free(buf);
  
}
std::shared_ptr<HDVector> InMemoryDataSet::getHDVecByIndex(const int & index){
  std::shared_ptr<HDVector> vec = this->m_data.at(index);
  std::cout << "req ind : "<< index << "del ind : " <<(*vec)[0];
  return vec;
}

std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> InMemoryDataSet::getNHDVectorsFromIndex(const int  &index,const int & n)   {
  std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> vec = std::make_unique<std::vector<std::shared_ptr<HDVector>>>( this->m_data.begin() + index , this->m_data.begin() +n );
  return vec;
 }
