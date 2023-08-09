#include <vector>
#include <filesystem>
#include <fstream>
#include "HDVector.hpp"
#ifndef DATASET
#define DATASET

namespace fs = std::filesystem;
class DataSet{
    protected:
        long long int n;
        long long int dimentions;
    public:
        DataSet() = default;
        DataSet(const DataSet&) = delete;
        virtual const HDVector &  getHDVecByIndex(const int  &index) const = 0;
        virtual const std::vector<HDVector*> & getNHDVectorsFromIndex(const int  &index,const int & n) const = 0;
        const int getN() const {
            return  this->n;
        }
        const int getDimentions() const {
            return  this->dimentions;
        }

};

class FileDataSet :public DataSet{
    private:
        std::fstream m_file;
    public:
        FileDataSet(fs::path path );
        const HDVector & getHDVecByIndex(const int  &index);
        const std::vector<HDVector*> &getNHDVectorsFromIndex(const int  &index,const int & n);
        using DataSet::getN;
        using DataSet::getDimentions;
};

class InMemoryDataSet :public DataSet{
    private:
        std::vector<HDVector*>* m_data;
        std::fstream m_file;
        std::vector<HDVector *> * readDataFromFile();
    public:
        InMemoryDataSet(fs::path path );
        const HDVector & getHDVecByIndex(const int  &index) const;
        const std::vector<HDVector*> & getNHDVectorsFromIndex(const int  &index,const int & n) const ;
        using DataSet::getN;
        using DataSet::getDimentions;
};
#endif
