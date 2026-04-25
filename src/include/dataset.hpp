#ifndef DATASET
#define DATASET

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "Vector.hpp"
#include "node_types.hpp"

#define rowsize(x) (x*sizeof(float) + sizeof(int64_t))
namespace fs = std::filesystem;

struct RecordView {
  int64_t recordId;
  std::shared_ptr<Vector> vector;
};

class DataSet {
protected:
  uint64_t n = 0;
  uint64_t dimensions = 0;
  uint64_t storedDimensions = 0;
  std::fstream m_file;

public:
  DataSet() = default;
  DataSet(const DataSet &) = delete;
  virtual ~DataSet() = default;
  virtual RecordView getRecordViewByIndex(uint64_t index) = 0;
  virtual std::unique_ptr<std::vector<RecordView>>
  getNRecordViewsFromIndex(uint64_t index, uint64_t n) = 0;
  virtual std::unique_ptr<std::vector<std::shared_ptr<Vector>>>
  getNVectorsFromIndex(uint64_t index, uint64_t n) = 0;
  uint64_t getN() const { return this->n; }
  uint64_t getDimensions() const { return this->dimensions; }
  uint64_t getStoredDimensions() const { return this->storedDimensions; }
/*   virtual float distance(const int &vector1, const int &vector2) = 0; */
};

class FileDataSet : public DataSet {
public:
  FileDataSet(fs::path path);
  RecordView getRecordViewByIndex(uint64_t index);
  std::unique_ptr<std::vector<RecordView>>
  getNRecordViewsFromIndex(uint64_t index, uint64_t n);
  std::unique_ptr<std::vector<std::shared_ptr<Vector>>>
  getNVectorsFromIndex(uint64_t index, uint64_t n);
  using DataSet::getDimensions;
  using DataSet::getN;
/*   float distance(const int &vector1, const int &vector2); */
};

class InMemoryDataSet : public DataSet {
private:
  std::vector<RecordView> m_records;
  void readDataFromFile();

public:
  InMemoryDataSet(fs::path path);
  RecordView getRecordViewByIndex(uint64_t index);
  std::unique_ptr<std::vector<RecordView>>
  getNRecordViewsFromIndex(uint64_t index, uint64_t n);
  std::unique_ptr<std::vector<std::shared_ptr<Vector>>>
  getNVectorsFromIndex(uint64_t index, uint64_t n);
  using DataSet::getDimensions;
  using DataSet::getN;
/*   float distance(const int &vector1, const int &vector2); */
};

#endif  // DATASET
