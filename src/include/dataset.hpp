#include "HDVector.hpp"
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#ifndef DATASET
#define DATASET

namespace fs = std::filesystem;

struct RecordView {
  long long recordId;
  std::shared_ptr<HDVector> vector;
};

class DataSet {
protected:
  long long int n;
  long long int dimentions;
  long long int storedDimentions;
  std::fstream m_file;

public:
  DataSet() = default;
  DataSet(const DataSet &) = delete;
  virtual ~DataSet() = default;
  virtual RecordView getRecordViewByIndex(const int &index) = 0;
  virtual std::unique_ptr<std::vector<RecordView>>
  getNRecordViewsFromIndex(const int &index, const int &n) = 0;
  virtual std::unique_ptr<std::vector<std::shared_ptr<HDVector>>>
  getNHDVectorsFromIndex(const int &index, const int &n) = 0;
  const int getN() const { return this->n; }
  const int getDimentions() const { return this->dimentions; }
/*   virtual float distance(const int &vector1, const int &vector2) = 0; */
};

class FileDataSet : public DataSet {
public:
  FileDataSet(fs::path path);
  RecordView getRecordViewByIndex(const int &index);
  std::unique_ptr<std::vector<RecordView>>
  getNRecordViewsFromIndex(const int &index, const int &n);
  std::unique_ptr<std::vector<std::shared_ptr<HDVector>>>
  getNHDVectorsFromIndex(const int &index, const int &n);
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
  RecordView getRecordViewByIndex(const int &index);
  std::unique_ptr<std::vector<RecordView>>
  getNRecordViewsFromIndex(const int &index, const int &n);
  std::unique_ptr<std::vector<std::shared_ptr<HDVector>>>
  getNHDVectorsFromIndex(const int &index, const int &n);
  using DataSet::getDimentions;
  using DataSet::getN;
/*   float distance(const int &vector1, const int &vector2); */
};
#endif
