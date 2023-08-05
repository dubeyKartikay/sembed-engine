#include <filesystem>
#include <fstream>
#include "dataset.hpp"
#include <vector>
#include <algorithm>
namespace fs = std::filesystem;
const int STARTING_HEADER_OFFSET = 8;
bool isValidPath(const std::string& path) {
        return fs::exists(path);
}
FileDataSet::FileDataSet(fs::path path){
        if(!isValidPath(path)){
            throw std::invalid_argument("The path does not exist");
        }
        m_file.open(path);
        if(!m_file.is_open()){
            throw std::runtime_error("could not open the file provided");
        }
        m_file.read((char*)&this->n,sizeof(this->n));
        m_file.read((char*)&this->dimentions,sizeof(this->dimentions));
    }



std::vector<float> * FileDataSet::getVecByIndex(const int  &index){

    std::vector<float> * vector  = new std::vector<float>(dimentions,0);
    m_file.seekg(STARTING_HEADER_OFFSET + index*dimentions*sizeof(float));
    m_file.read(reinterpret_cast<char*>(vector->data()),dimentions*sizeof(float));
    return vector;

}

std::vector<std::vector<float>*>* FileDataSet::getNVectorsFromIndex(const int &index,const int & n){
    std::vector<std::vector<float>*> * vec = new std::vector<std::vector<float>*>;
    vec->reserve(n);
    m_file.seekg(STARTING_HEADER_OFFSET + index*dimentions*sizeof(float));
    float * buffer  = (float *) malloc(n*dimentions*sizeof(float));
    if(buffer == NULL){
        throw std::runtime_error("Not enough memory");
    }
    m_file.read(reinterpret_cast<char*>(buffer),n*dimentions*sizeof(float));
    for (size_t i = 0; i < n; i++)
    {
        std::vector<float> * vector_floats = new std::vector<float>();
        std::copy_n(buffer + i*dimentions,dimentions,vector_floats->begin());
        vec->push_back(vector_floats);
    }
    free(buffer);
    
    return vec;
}


