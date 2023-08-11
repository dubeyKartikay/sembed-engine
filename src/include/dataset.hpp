#include <vector>
#include <filesystem>
#include <fstream>
#include "HDVector.hpp"
#include <memory>
#ifndef DATASET
#define DATASET

namespace fs = std::filesystem;
class DataSet{
    protected:
        long long int n;
        long long int dimentions;
        std::fstream m_file;
    public:
        DataSet() = default;
        DataSet(const DataSet&) = delete;
        virtual const HDVector &  getHDVecByIndex(const int  &index) const = 0;
        virtual std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> getNHDVectorsFromIndex(const int  &index,const int & n);
        const int getN() const {
            return  this->n;
        }
        const int getDimentions() const {
            return  this->dimentions;
        }

};

class FileDataSet :public DataSet{
    public:
        FileDataSet(fs::path path );
        const HDVector & getHDVecByIndex(const int  &index);
        std::unique_ptr<std::vector<std::shared_ptr<HDVector>>>getNHDVectorsFromIndex(const int  &index,const int & n);
        using DataSet::getN;
        using DataSet::getDimentions;
};

class InMemoryDataSet :public DataSet{
    private:
        std::vector<std::shared_ptr<HDVector>>   m_data;
        void readDataFromFile(std::vector<std::shared_ptr<HDVector>> & m_data);
    public:
        InMemoryDataSet(fs::path path );
        const HDVector & getHDVecByIndex(const int  &index) const;
        std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> getNHDVectorsFromIndex(const int  &index,const int & n);
        using DataSet::getN;
        using DataSet::getDimentions;
};
#endif
