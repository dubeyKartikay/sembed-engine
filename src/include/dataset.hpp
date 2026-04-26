#ifndef DATASET
#define DATASET

#include <cstdint>
#include <filesystem>
#include <vector>

#include "armadillo"
#include "vector_view.hpp"

namespace fs = std::filesystem;

struct RecordView {
  int64_t recordId = 0;
  FloatVectorView values;
};

class DataSet {
protected:
  uint64_t m_n = 0;
  uint64_t m_dimensions = 0;
  uint64_t m_storedDimensions = 0;

public:
  DataSet() = default;
  DataSet(const DataSet &) = delete;
  DataSet &operator=(const DataSet &) = delete;
  virtual ~DataSet() = default;

  virtual RecordView getRecordViewByIndex(uint64_t index) const = 0;
  virtual std::vector<RecordView> getRecordViewsFromIndex(
      uint64_t index, uint64_t count) const = 0;

  uint64_t getN() const { return m_n; }
  uint64_t getDimensions() const { return m_dimensions; }
  uint64_t getStoredDimensions() const { return m_storedDimensions; }
};

class FlatDataSet : public DataSet {
private:
  std::vector<int64_t> m_recordIds;
  arma::fmat m_matrix;

public:
  explicit FlatDataSet(fs::path path);
  RecordView getRecordViewByIndex(uint64_t index) const override;
  std::vector<RecordView> getRecordViewsFromIndex(
      uint64_t index, uint64_t count) const override;
};

#endif  // DATASET
