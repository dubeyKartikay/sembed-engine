#include "dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.hpp"

FlatDataSet::FlatDataSet(fs::path path) {
  if (!isValidFile(path.string())) {
    throw std::invalid_argument("dataset path must refer to a readable file");
  }

  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("could not open the file provided");
  }

  int64_t rawN = 0;
  int64_t rawStoredDimensions = 0;
  file.read(reinterpret_cast<char *>(&rawN), sizeof(rawN));
  file.read(reinterpret_cast<char *>(&rawStoredDimensions),
            sizeof(rawStoredDimensions));
  if (!file) {
    throw std::runtime_error("failed to read dataset header");
  }
  if (rawN < 0) {
    throw std::runtime_error("dataset record count must be non-negative");
  }
  if (rawStoredDimensions <= 1) {
    throw std::runtime_error(
        "dataset vectors must include at least a record id and some data");
  }

  m_n = static_cast<uint64_t>(rawN);
  m_storedDimensions = static_cast<uint64_t>(rawStoredDimensions);
  m_dimensions = m_storedDimensions - 1;

  if (m_dimensions != 0 &&
      m_n > std::numeric_limits<size_t>::max() / m_dimensions) {
    throw std::runtime_error("dataset is too large to load into memory");
  }

  m_recordIds.resize(static_cast<size_t>(m_n));
  m_values.resize(static_cast<size_t>(m_n * m_dimensions));

  for (uint64_t row = 0; row < m_n; ++row) {
    file.read(reinterpret_cast<char *>(
                  &m_recordIds.at(static_cast<size_t>(row))),
              sizeof(int64_t));
    float *rowData =
        m_values.data() + static_cast<size_t>(row * m_dimensions);
    file.read(reinterpret_cast<char *>(rowData),
              static_cast<std::streamsize>(m_dimensions * sizeof(float)));
    if (!file) {
      throw std::runtime_error("failed to read dataset into memory");
    }
  }
}

RecordView FlatDataSet::getRecordViewByIndex(uint64_t index) const {
  if (index >= m_n) {
    throw std::out_of_range("record index is outside dataset bounds");
  }

  const size_t offset = static_cast<size_t>(index * m_dimensions);
  return {m_recordIds.at(static_cast<size_t>(index)),
          FloatVectorView(m_values.data() + offset, m_dimensions)};
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
