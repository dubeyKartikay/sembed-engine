#include "dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.hpp"

namespace {

size_t checkedValueCount(uint64_t records, uint64_t dimensions) {
  if (dimensions != 0 &&
      records > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
                    dimensions) {
    throw std::runtime_error("dataset is too large to load into memory");
  }
  return static_cast<size_t>(records * dimensions);
}

void readDataSet(std::istream& input, uint64_t& records, uint64_t& dimensions,
                 uint64_t& storedDimensions,
                 std::vector<int64_t>& recordIds,
                 std::vector<float>& vectors) {
  int64_t rawRecords = 0;
  int64_t rawStoredDimensions = 0;
  input.read(reinterpret_cast<char*>(&rawRecords), sizeof(rawRecords));
  input.read(reinterpret_cast<char*>(&rawStoredDimensions),
             sizeof(rawStoredDimensions));
  if (!input) {
    throw std::runtime_error("failed to read dataset header");
  }
  if (rawRecords < 0) {
    throw std::runtime_error("dataset record count must be non-negative");
  }
  if (rawStoredDimensions <= 1) {
    throw std::runtime_error(
        "dataset vectors must include at least a record id and some data");
  }

  records = static_cast<uint64_t>(rawRecords);
  storedDimensions = static_cast<uint64_t>(rawStoredDimensions);
  dimensions = storedDimensions - 1;
  recordIds.resize(static_cast<size_t>(records));
  vectors.resize(checkedValueCount(records, dimensions));

  input.read(reinterpret_cast<char*>(recordIds.data()),
             static_cast<std::streamsize>(sizeof(int64_t) * records));
  input.read(reinterpret_cast<char*>(vectors.data()),
             static_cast<std::streamsize>(sizeof(float) * vectors.size()));
  if (!input) {
    throw std::runtime_error("failed to read dataset data");
  }
}

}  // namespace

FlatDataSet::FlatDataSet(fs::path path) {
  if (!isValidFile(path.string())) {
    throw std::invalid_argument("dataset path must refer to a readable file");
  }

  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("could not open the file provided");
  }
  readDataSet(input, m_n, m_dimensions, m_storedDimensions, m_recordIds,
              m_vectors);

  char trailing = '\0';
  if (input.read(&trailing, 1)) {
    throw std::runtime_error("dataset file contains trailing data");
  }
  if (!input.eof()) {
    throw std::runtime_error("failed to validate dataset file size");
  }
}

FlatDataSet::FlatDataSet(std::istream& input) {
  readDataSet(input, m_n, m_dimensions, m_storedDimensions, m_recordIds,
              m_vectors);
}

RecordView FlatDataSet::getRecordViewByIndex(uint64_t index) const {
  if (index >= m_n) {
    throw std::out_of_range("record index is outside dataset bounds");
  }
  const size_t offset = static_cast<size_t>(index * m_dimensions);
  return {m_recordIds.at(static_cast<size_t>(index)),
          FloatVectorView(m_vectors.data() + offset, m_dimensions)};
}

std::vector<RecordView> FlatDataSet::getRecordViewsFromIndex(
    uint64_t index, uint64_t count) const {
  if (index > m_n || count > m_n - index) {
    throw std::out_of_range("record range is outside dataset bounds");
  }

  std::vector<RecordView> records;
  records.reserve(static_cast<size_t>(count));
  for (uint64_t offset = 0; offset < count; ++offset) {
    records.push_back(getRecordViewByIndex(index + offset));
  }
  return records;
}

void FlatDataSet::addVector(int64_t recordId, const float* vector,
                            uint64_t dimensions) {
  if (dimensions != m_dimensions) {
    throw std::invalid_argument("vector dimensions must match the dataset");
  }
  if (dimensions != 0 && vector == nullptr) {
    throw std::invalid_argument("vector data must not be null");
  }

  m_recordIds.push_back(recordId);
  m_vectors.insert(m_vectors.end(), vector, vector + dimensions);
  ++m_n;
}

FlatDataSet::FlatDataSet(uint64_t dimensions) {
  m_dimensions = dimensions;
  m_n = 0;
  m_storedDimensions = m_dimensions + 1;
}

FlatDataSet::FlatDataSet(uint64_t dimensions, uint64_t capacity) {
  m_dimensions = dimensions;
  m_n = capacity;
  m_storedDimensions = m_dimensions + 1;
  m_recordIds.resize(static_cast<size_t>(capacity));
  m_vectors.resize(checkedValueCount(capacity, dimensions));
}

void FlatDataSet::setVectorByIndex(uint64_t index, int64_t recordId,
                                   const float* vector, uint64_t dimensions) {
  if (dimensions != m_dimensions) {
    throw std::invalid_argument("vector dimensions must match the dataset");
  }
  if (index >= m_n) {
    throw std::out_of_range("record index is outside dataset bounds");
  }
  if (dimensions != 0 && vector == nullptr) {
    throw std::invalid_argument("vector data must not be null");
  }

  m_recordIds[static_cast<size_t>(index)] = recordId;
  const size_t offset = static_cast<size_t>(index * dimensions);
  std::copy(vector, vector + dimensions, m_vectors.data() + offset);
}

void FlatDataSet::save(std::ostream& output) const {
  if (m_n > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      m_storedDimensions >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("dataset shape cannot be serialized");
  }
  const int64_t records = static_cast<int64_t>(m_n);
  const int64_t storedDimensions = static_cast<int64_t>(m_storedDimensions);
  output.write(reinterpret_cast<const char*>(&records), sizeof(records));
  output.write(reinterpret_cast<const char*>(&storedDimensions),
               sizeof(storedDimensions));
  output.write(reinterpret_cast<const char*>(m_recordIds.data()),
               static_cast<std::streamsize>(sizeof(int64_t) * m_recordIds.size()));
  output.write(reinterpret_cast<const char*>(m_vectors.data()),
               static_cast<std::streamsize>(sizeof(float) * m_vectors.size()));
  if (!output) {
    throw std::runtime_error("failed to write dataset data");
  }
}
