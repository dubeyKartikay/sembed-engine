#include "HDVector.hpp"
#include "dataset.hpp"
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>
namespace fs = std::filesystem;
namespace {
constexpr std::streamoff STARTING_HEADER_OFFSET =
    static_cast<std::streamoff>(2 * sizeof(uint64_t));
}
bool isValidPath(const std::string &path) { return fs::exists(path); }
bool isValidFile(const std::string &path) { return isValidPath(path) && !fs::is_directory(path); }
FileDataSet::FileDataSet(fs::path path) {
  if (!isValidFile(path)) {
    throw std::invalid_argument("The path does not exist");
  }
  m_file.open(path, std::ios::binary | std::ios::in);
  if (!m_file.is_open()) {
    throw std::runtime_error("could not open the file provided");
  }
  int64_t raw_n = 0;
  int64_t raw_stored_dimentions = 0;
  m_file.read(reinterpret_cast<char *>(&raw_n), sizeof(raw_n));
  m_file.read(reinterpret_cast<char *>(&raw_stored_dimentions),
              sizeof(raw_stored_dimentions));
  if (raw_n < 0) {
    throw std::runtime_error("dataset record count must be non-negative");
  }
  if (raw_stored_dimentions <= 1) {
    throw std::runtime_error(
        "dataset vectors must include at least a record id and some data");
  }
  this->n = static_cast<uint64_t>(raw_n);
  this->storedDimentions = static_cast<uint64_t>(raw_stored_dimentions);
  this->dimentions = this->storedDimentions - 1;
}

RecordView FileDataSet::getRecordViewByIndex(uint64_t index) {
  if (index >= this->getN()) {
    throw std::out_of_range("record index is outside dataset bounds");
  }

  std::vector<char> buffer(rowsize(this->dimentions), 0);
  std::shared_ptr<HDVector> vector =
      std::make_shared<HDVector>(this->getDimentions());
  m_file.clear();
  m_file.seekg(STARTING_HEADER_OFFSET +
               static_cast<std::streamoff>(index * rowsize(this->dimentions)));
  m_file.read(reinterpret_cast<char *>(buffer.data()),
              static_cast<std::streamsize>(rowsize(this->dimentions)));
  if (!m_file) {
    throw std::runtime_error("failed to read record from dataset");
  }

  int64_t id = 0;
  std::memcpy(&id, buffer.data(),
              sizeof(int64_t));
  std::memcpy(vector->getDataPointer(),
              buffer.data() + sizeof(int64_t),
              this->getDimentions() * sizeof(float));
  return {id, vector};
}

std::unique_ptr<std::vector<RecordView>>
FileDataSet::getNRecordViewsFromIndex(uint64_t index, uint64_t n) {
  const uint64_t limit = this->getN();
  if (index > limit || n > limit - index) {
    throw std::out_of_range("record range is outside dataset bounds");
  }

  std::unique_ptr<std::vector<RecordView>> records =
      std::make_unique<std::vector<RecordView>>();
  records->reserve(static_cast<size_t>(n));

  m_file.clear();
  m_file.seekg(STARTING_HEADER_OFFSET +
               static_cast<std::streamoff>(index * rowsize(this->dimentions)));

  std::vector<char> buffer(static_cast<size_t>(n) * rowsize(this->dimentions), 0);
  m_file.read(buffer.data(),
              static_cast<std::streamsize>(buffer.size()));
  if (!m_file) {
    throw std::runtime_error("failed to read record range from dataset");
  }

  for (uint64_t i = 0; i < n; ++i) {
    std::shared_ptr<HDVector> vector_floats =
        std::make_shared<HDVector>(dimentions);
    int64_t id = 0;
    const char *record = buffer.data() + static_cast<size_t>(i) * rowsize(this->dimentions);
    std::memcpy(&id, record, sizeof(int64_t));
    std::memcpy(vector_floats->getDataPointer(), record + sizeof(int64_t),
                this->getDimentions() * sizeof(float));
    records->push_back({id, vector_floats});
  }

  return records;
}

std::unique_ptr<std::vector<std::shared_ptr<HDVector>>>
FileDataSet::getNHDVectorsFromIndex(uint64_t index, uint64_t n) {
  auto records = getNRecordViewsFromIndex(index, n);
  std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> vec =
      std::make_unique<std::vector<std::shared_ptr<HDVector>>>();
  vec->reserve(records->size());
  for (const RecordView &record : *records) {
    vec->push_back(record.vector);
  }
  return vec;
}
