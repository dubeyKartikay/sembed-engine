#include <vector>
#include <filesystem>
#include <fstream>

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
        virtual std::vector<float> * getVecByIndex(const int  &index) = 0;
        virtual std::vector<std::vector<float>*>* getNVectorsFromIndex(const int  &index,const int & n) = 0;
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
        std::vector<float> * getVecByIndex(const int  &index);
        std::vector<std::vector<float>*>* getNVectorsFromIndex(const int  &index,const int & n);
        using DataSet::getN;
        using DataSet::getDimentions;
};

class InMemoryDataSet :public DataSet{
    private:
        std::vector<std::vector<float>*>* m_data;
    public:
        InMemoryDataSet(fs::path path );
        std::vector<float> * getVecByIndex(const int  &index);
        std::vector<std::vector<float>*>* getNVectorsFromIndex(const int  &index,const int & n);
        using DataSet::getN;
        using DataSet::getDimentions;
};
#endif