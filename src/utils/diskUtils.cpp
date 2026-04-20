#include "HDVector.hpp"
#include "dataset.hpp"
#include <cstdint>
#include <algorithm>
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
  if (raw_stored_dimentions < 1) {
    throw std::runtime_error(
        "dataset vectors must include at least a record id");
  }
  this->n = static_cast<uint64_t>(raw_n);
  this->storedDimentions = static_cast<uint64_t>(raw_stored_dimentions);
  this->dimentions = this->storedDimentions - 1;
}

RecordView FileDataSet::getRecordViewByIndex(int64_t index) {
  if (index < 0 || static_cast<uint64_t>(index) >= this->getN()) {
    throw std::out_of_range("record index is outside dataset bounds");
  }

  std::vector<float> buffer(static_cast<size_t>(this->storedDimentions), 0.0f);
  std::shared_ptr<HDVector> vector =
      std::make_shared<HDVector>(this->getDimentions());
  m_file.clear();
  m_file.seekg(STARTING_HEADER_OFFSET +
               static_cast<std::streamoff>(index * static_cast<int64_t>(storedDimentions) *
                                           static_cast<int64_t>(sizeof(float))));
  m_file.read(reinterpret_cast<char *>(buffer.data()),
              static_cast<std::streamsize>(storedDimentions * sizeof(float)));
  if (!m_file) {
    throw std::runtime_error("failed to read record from dataset");
  }
  std::copy_n(buffer.data() + 1, this->getDimentions(),
              vector->getDataPointer());
  return {static_cast<int64_t>(buffer[0]), vector};
}

std::unique_ptr<std::vector<RecordView>>
FileDataSet::getNRecordViewsFromIndex(int64_t index, int64_t n) {
  if (index < 0 || n < 0 ||
      static_cast<uint64_t>(index) + static_cast<uint64_t>(n) > this->getN()) {
    throw std::out_of_range("record range is outside dataset bounds");
  }

  std::unique_ptr<std::vector<RecordView>> records =
      std::make_unique<std::vector<RecordView>>();
  records->reserve(static_cast<size_t>(n));

  m_file.clear();
  m_file.seekg(STARTING_HEADER_OFFSET +
               static_cast<std::streamoff>(index * static_cast<int64_t>(storedDimentions) *
                                           static_cast<int64_t>(sizeof(float))));

  std::vector<float> buffer(static_cast<size_t>(n * static_cast<int64_t>(storedDimentions)),
                            0.0f);
  m_file.read(reinterpret_cast<char *>(buffer.data()),
              static_cast<std::streamsize>(n * static_cast<int64_t>(storedDimentions) *
                                           static_cast<int64_t>(sizeof(float))));
  if (!m_file) {
    throw std::runtime_error("failed to read record range from dataset");
  }

  for (int64_t i = 0; i < n; ++i) {
    std::shared_ptr<HDVector> vector_floats =
        std::make_shared<HDVector>(dimentions);
    std::copy_n(buffer.data() + i * storedDimentions + 1, dimentions,
                vector_floats->getDataPointer());
    records->push_back(
        {static_cast<int64_t>(buffer[i * storedDimentions]), vector_floats});
  }

  return records;
}

std::unique_ptr<std::vector<std::shared_ptr<HDVector>>>
FileDataSet::getNHDVectorsFromIndex(int64_t index, int64_t n) {
  auto records = getNRecordViewsFromIndex(index, n);
  std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> vec =
      std::make_unique<std::vector<std::shared_ptr<HDVector>>>();
  vec->reserve(records->size());
  for (const RecordView &record : *records) {
    vec->push_back(record.vector);
  }
  return vec;
}
