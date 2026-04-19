#include <filesystem>
#include <memory>
#include <fstream>
#include "dataset.hpp"
#include "HDVector.hpp"
#include <vector>
#include <algorithm>
namespace fs = std::filesystem;
#define STARTING_HEADER_OFFSET 2*sizeof(long long int) 
bool isValidPath(const std::string& path) {
        return fs::exists(path);
}
FileDataSet::FileDataSet(fs::path path){
        if(!isValidPath(path)){
            throw std::invalid_argument("The path does not exist");
        }
        m_file.open(path, std::ios::binary | std::ios::in);
        if(!m_file.is_open()){
            throw std::runtime_error("could not open the file provided");
        }
        m_file.read((char*)&this->n,sizeof(this->n));
        m_file.read((char*)&this->storedDimentions,sizeof(this->storedDimentions));
        if (this->storedDimentions < 1) {
            throw std::runtime_error("dataset vectors must include at least a record id");
        }
        this->dimentions = this->storedDimentions - 1;
    }



RecordView FileDataSet::getRecordViewByIndex(const int  &index){
    if (index < 0 || index >= this->getN()) {
        throw std::out_of_range("record index is outside dataset bounds");
    }

    std::vector<float> buffer(this->storedDimentions, 0.0f);
    std::shared_ptr<HDVector> vector  = std::make_shared<HDVector>(this->getDimentions());
    m_file.clear();
    m_file.seekg(STARTING_HEADER_OFFSET + index * storedDimentions * sizeof(float));
    m_file.read(reinterpret_cast<char*>(buffer.data()),
                storedDimentions * sizeof(float));
    if (!m_file) {
        throw std::runtime_error("failed to read record from dataset");
    }
    std::copy_n(buffer.data() + 1, this->getDimentions(), vector->getDataPointer());
    return {static_cast<long long>(buffer[0]), vector};
}

std::unique_ptr<std::vector<RecordView>>
FileDataSet::getNRecordViewsFromIndex(const int &index,const int & n){
    if (index < 0 || n < 0 || index + n > this->getN()) {
        throw std::out_of_range("record range is outside dataset bounds");
    }

    std::unique_ptr<std::vector<RecordView>> records =
        std::make_unique<std::vector<RecordView>>();
    records->reserve(n);

    m_file.clear();
    m_file.seekg(STARTING_HEADER_OFFSET + index * storedDimentions * sizeof(float));

    std::vector<float> buffer(n * storedDimentions, 0.0f);
    m_file.read(reinterpret_cast<char*>(buffer.data()),
                n * storedDimentions * sizeof(float));
    if (!m_file) {
        throw std::runtime_error("failed to read record range from dataset");
    }

    for (int i = 0; i < n; i++) {
        std::shared_ptr<HDVector> vector_floats =
            std::make_shared<HDVector>(dimentions);
        std::copy_n(buffer.data() + i * storedDimentions + 1, dimentions,
                    vector_floats->getDataPointer());
        records->push_back(
            {static_cast<long long>(buffer[i * storedDimentions]), vector_floats});
    }

    return records;
}

std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> FileDataSet::getNHDVectorsFromIndex(const int &index,const int & n){
    auto records = getNRecordViewsFromIndex(index, n);
    std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> vec =
        std::make_unique<std::vector<std::shared_ptr<HDVector>>>();
    vec->reserve(records->size());
    for (const RecordView &record : *records) {
        vec->push_back(record.vector);
    }
    return vec;
}
