#include "dataset.hpp"
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
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
  int64_t raw_n = 0;
  int64_t raw_stored_dimentions = 0;
  m_file.read(reinterpret_cast<char*>(&raw_n), sizeof(raw_n));
  m_file.read(reinterpret_cast<char*>(&raw_stored_dimentions),
              sizeof(raw_stored_dimentions));
  if (raw_n < 0) {
      throw std::runtime_error("dataset record count must be non-negative");
  }
  if (raw_stored_dimentions <= 1) {
      throw std::runtime_error("dataset vectors must include at least a record id and some data");
  }
  this->n = static_cast<uint64_t>(raw_n);
  this->storedDimentions = static_cast<uint64_t>(raw_stored_dimentions);
  this->dimentions = this->storedDimentions - 1;
  readDataFromFile(); // reads data to m_data
  m_file.close();

}
void InMemoryDataSet::readDataFromFile(){
  if (this->storedDimentions != 0 &&
      this->getN() > std::numeric_limits<size_t>::max() / this->storedDimentions) {
    throw std::runtime_error("dataset is too large to load into memory");
  }

  const size_t total_floats =
      static_cast<size_t>(this->dimentions * this->getN());

  std::vector<char*> buf(this->getN()*sizeof(int64_t) +total_floats * sizeof(float));

  m_file.read(reinterpret_cast<char *>(buf.data()),
              static_cast<std::streamsize>(this->getN()*sizeof(int64_t) +total_floats * sizeof(float)));

  if (!m_file) {
    throw std::runtime_error("failed to read dataset into memory");
  }
  m_records.reserve(static_cast<size_t>(this->getN()));
  for (uint64_t i = 0; i < this->getN(); ++i) {
    std::shared_ptr<HDVector> hdv = std::make_shared<HDVector>(dimentions);
    std::copy_n(buf.data() + i * this->storedDimentions + 1, this->getDimentions(),
                hdv->getDataPointer());
    m_records.push_back({static_cast<int64_t>(buf[i * this->storedDimentions]),
                         hdv});
  }
}
RecordView InMemoryDataSet::getRecordViewByIndex(int64_t index){
  if (index < 0) {
    throw std::out_of_range("record index is outside dataset bounds");
  }
  return this->m_records.at(static_cast<size_t>(index));
}

std::unique_ptr<std::vector<RecordView>>
InMemoryDataSet::getNRecordViewsFromIndex(int64_t index, int64_t n) {
  if (index < 0 || n < 0 ||
      static_cast<uint64_t>(index) + static_cast<uint64_t>(n) > m_records.size()) {
    throw std::out_of_range("record range is outside dataset bounds");
  }
  return std::make_unique<std::vector<RecordView>>(
      this->m_records.begin() + index, this->m_records.begin() + index + n);
}

std::unique_ptr<std::vector<std::shared_ptr<HDVector>>>
InMemoryDataSet::getNHDVectorsFromIndex(int64_t index, int64_t n) {
  auto records = getNRecordViewsFromIndex(index, n);
  std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> vec =
      std::make_unique<std::vector<std::shared_ptr<HDVector>>>();
  vec->reserve(records->size());
  for (const RecordView &record : *records) {
    vec->push_back(record.vector);
  }
  return vec;
 }
