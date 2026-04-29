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
  DataSet(DataSet &&) = default;
  DataSet &operator=(DataSet &&) = default;
  virtual ~DataSet() = default;

  virtual RecordView getRecordViewByIndex(uint64_t index) const = 0;
  virtual std::vector<RecordView> getRecordViewsFromIndex(
      uint64_t index, uint64_t count) const = 0;

  uint64_t getN() const { return m_n; }
  uint64_t getDimensions() const { return m_dimensions; }
  uint64_t getStoredDimensions() const { return m_storedDimensions; }
  virtual void addVector(int64_t recordId, const float* vector,
                         uint64_t dimensions) = 0;
  virtual float *data() = 0;
};

class FlatDataSet : public DataSet {
private:
  std::vector<int64_t> m_recordIds;
  arma::fmat m_matrix;

public:
  explicit FlatDataSet(fs::path path);
  FlatDataSet(uint64_t dimensions);
  FlatDataSet(uint64_t dimensions, uint64_t capacity);
  FlatDataSet(const FlatDataSet &) = delete;
  FlatDataSet &operator=(const FlatDataSet &) = delete;
  FlatDataSet(FlatDataSet &&) = default;
  FlatDataSet &operator=(FlatDataSet &&) = default;
  float *data() override { return m_matrix.memptr(); }
  RecordView getRecordViewByIndex(uint64_t index) const override;
  void addVector(int64_t recordId, const float* vector,
                 uint64_t dimensions) override;
  void setVectorByIndex(uint64_t index, int64_t recordId, const float* vector,
                        uint64_t dimensions);
  std::vector<RecordView> getRecordViewsFromIndex(
      uint64_t index, uint64_t count) const override;
};

#endif  // DATASET
