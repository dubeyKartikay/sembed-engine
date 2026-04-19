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
  m_file.open(path, std::ios::binary | std::ios::in);
  if(!m_file.is_open()){
      throw std::runtime_error("could not open the file provided");
  }
  m_file.read((char*)&this->n,sizeof(this->n));
  m_file.read((char*)&this->storedDimentions,sizeof(this->storedDimentions));
  if (this->storedDimentions < 1) {
      throw std::runtime_error("dataset vectors must include at least a record id");
  }
  this->dimentions = this->storedDimentions - 1;
  readDataFromFile(); // reads data to m_data
  m_file.close();

}
void InMemoryDataSet::readDataFromFile(){
  float * buf = (float *)malloc(this->storedDimentions * this->getN() * sizeof(float));
  if(buf == NULL){
    throw std::runtime_error("Cannot allocated the required memoory to load the dataset into memory");
  }
  m_file.read(reinterpret_cast<char *>(buf), this->storedDimentions * this->getN() * sizeof(float));
  if (!m_file) {
    free(buf);
    throw std::runtime_error("failed to read dataset into memory");
  }
  m_records.reserve(this->getN());
  for(int i = 0;i < this->getN(); i++){
    std::shared_ptr<HDVector> hdv = std::make_shared<HDVector>(dimentions);
    std::copy_n(buf + i * this->storedDimentions + 1, this->getDimentions(),
                hdv->getDataPointer());
    m_records.push_back({static_cast<long long>(buf[i * this->storedDimentions]),
                         hdv});
  }
  free(buf);
  
}
RecordView InMemoryDataSet::getRecordViewByIndex(const int & index){
  return this->m_records.at(index);
}

std::unique_ptr<std::vector<RecordView>>
InMemoryDataSet::getNRecordViewsFromIndex(const int & index,const int & n) {
  if (index < 0 || n < 0 || index + n > static_cast<int>(m_records.size())) {
    throw std::out_of_range("record range is outside dataset bounds");
  }
  return std::make_unique<std::vector<RecordView>>(this->m_records.begin() + index,
                                                   this->m_records.begin() + index + n);
}

std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> InMemoryDataSet::getNHDVectorsFromIndex(const int & index,const int & n)   {
  auto records = getNRecordViewsFromIndex(index, n);
  std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> vec =
      std::make_unique<std::vector<std::shared_ptr<HDVector>>>();
  vec->reserve(records->size());
  for (const RecordView &record : *records) {
    vec->push_back(record.vector);
  }
  return vec;
 }
