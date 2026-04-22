#include "dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "HDVector.hpp"
#include "utils.hpp"

InMemoryDataSet::InMemoryDataSet(fs::path path) {
  if (!isValidFile(path.string())) {
      throw std::invalid_argument("dataset path must refer to a readable file");
  }
  m_file.open(path, std::ios::binary | std::ios::in);
  if (!m_file.is_open()) {
      throw std::runtime_error("could not open the file provided");
  }
  int64_t raw_n = 0;
  int64_t raw_stored_dimensions = 0;
  m_file.read(reinterpret_cast<char*>(&raw_n), sizeof(raw_n));
  m_file.read(reinterpret_cast<char*>(&raw_stored_dimensions),
              sizeof(raw_stored_dimensions));
  if (!m_file) {
      throw std::runtime_error("failed to read dataset header");
  }
  if (raw_n < 0) {
      throw std::runtime_error("dataset record count must be non-negative");
  }
  if (raw_stored_dimensions <= 1) {
      throw std::runtime_error("dataset vectors must include at least a record id and some data");
  }
  this->n = static_cast<uint64_t>(raw_n);
  this->storedDimensions = static_cast<uint64_t>(raw_stored_dimensions);
  this->dimensions = this->storedDimensions - 1;
  readDataFromFile();
  m_file.close();
}

void InMemoryDataSet::readDataFromFile() {
  if (rowsize(dimensions) != 0 &&
      getN() > std::numeric_limits<size_t>::max() / rowsize(dimensions)) {
    throw std::runtime_error("dataset is too large to load into memory");
  }

  std::vector<char> buffer(static_cast<size_t>(getN()) * rowsize(dimensions), 0);

  m_file.read(reinterpret_cast<char *>(buffer.data()),
              static_cast<std::streamsize>(buffer.size()));

  if (!m_file) {
    throw std::runtime_error("failed to read dataset into memory");
  }
  m_records.reserve(static_cast<size_t>(getN()));
  for (uint64_t i = 0; i < getN(); ++i) {
    std::shared_ptr<Vector> hdv = std::make_shared<HDVector>(dimensions);
    int64_t id = 0;
    std::memcpy(&id, buffer.data() + i * rowsize(dimensions),
                sizeof(int64_t));
    std::memcpy(hdv->getDataPointer(),
                buffer.data() + i * rowsize(dimensions) + sizeof(int64_t),
                getDimensions() * sizeof(float));
    m_records.push_back({id, hdv});
  }
}

RecordView InMemoryDataSet::getRecordViewByIndex(uint64_t index) {
  return m_records.at(static_cast<size_t>(index));
}

std::unique_ptr<std::vector<RecordView>>
InMemoryDataSet::getNRecordViewsFromIndex(uint64_t index, uint64_t n) {
  const uint64_t limit = m_records.size();
  if (index > limit || n > limit - index) {
    throw std::out_of_range("record range is outside dataset bounds");
  }
  return std::make_unique<std::vector<RecordView>>(
      this->m_records.begin() + static_cast<std::ptrdiff_t>(index),
      this->m_records.begin() + static_cast<std::ptrdiff_t>(index + n));
}

std::unique_ptr<std::vector<std::shared_ptr<Vector>>>
InMemoryDataSet::getNVectorsFromIndex(uint64_t index, uint64_t n) {
  auto records = getNRecordViewsFromIndex(index, n);
  std::unique_ptr<std::vector<std::shared_ptr<Vector>>> vec =
      std::make_unique<std::vector<std::shared_ptr<Vector>>>();
  vec->reserve(records->size());
  for (const RecordView &record : *records) {
    vec->push_back(record.vector);
  }
  return vec;
}
