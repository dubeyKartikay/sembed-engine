#include "HDVector.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include "node_types.hpp"
#ifndef DATASET
#define DATASET
#define rowsize(x) (x*sizeof(float) + sizeof(int64_t))
namespace fs = std::filesystem;

struct RecordView {
  int64_t recordId;
  std::shared_ptr<HDVector> vector;
};

class DataSet {
protected:
  uint64_t n;
  uint64_t dimentions;
  uint64_t storedDimentions;
  std::fstream m_file;

public:
  DataSet() = default;
  DataSet(const DataSet &) = delete;
  virtual ~DataSet() = default;
  virtual RecordView getRecordViewByIndex(uint64_t index) = 0;
  virtual std::unique_ptr<std::vector<RecordView>>
  getNRecordViewsFromIndex(uint64_t index, uint64_t n) = 0;
  virtual std::unique_ptr<std::vector<std::shared_ptr<HDVector>>>
  getNHDVectorsFromIndex(uint64_t index, uint64_t n) = 0;
  uint64_t getN() const { return this->n; }
  uint64_t getDimentions() const { return this->dimentions; }
/*   virtual float distance(const int &vector1, const int &vector2) = 0; */
};

class FileDataSet : public DataSet {
public:
  FileDataSet(fs::path path);
  RecordView getRecordViewByIndex(uint64_t index);
  std::unique_ptr<std::vector<RecordView>>
  getNRecordViewsFromIndex(uint64_t index, uint64_t n);
  std::unique_ptr<std::vector<std::shared_ptr<HDVector>>>
  getNHDVectorsFromIndex(uint64_t index, uint64_t n);
  using DataSet::getDimentions;
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
  std::unique_ptr<std::vector<std::shared_ptr<HDVector>>>
  getNHDVectorsFromIndex(uint64_t index, uint64_t n);
  using DataSet::getDimentions;
  using DataSet::getN;
/*   float distance(const int &vector1, const int &vector2); */
};
#endif
