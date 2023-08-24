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
        m_file.open(path);
        if(!m_file.is_open()){
            throw std::runtime_error("could not open the file provided");
        }
        m_file.read((char*)&this->n,sizeof(this->n));
        m_file.read((char*)&this->dimentions,sizeof(this->dimentions));
    }



std::shared_ptr<HDVector> FileDataSet::getHDVecByIndex(const int  &index){

    std::shared_ptr<HDVector> vector  = std::make_shared<HDVector>(this->getDimentions());
    m_file.seekg(STARTING_HEADER_OFFSET + index*dimentions*sizeof(float));
    m_file.read(reinterpret_cast<char*>(vector->getDataPointer()),dimentions*sizeof(float));
    return vector;

}

std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> FileDataSet::getNHDVectorsFromIndex(const int &index,const int & n){
  std::unique_ptr<std::vector<std::shared_ptr<HDVector>>> vec = std::make_unique<std::vector<std::shared_ptr<HDVector>>>();
    vec->reserve(n);
    
    m_file.seekg(STARTING_HEADER_OFFSET + index*dimentions*sizeof(float));
    
    float * buffer  = (float *) malloc(n*dimentions*sizeof(float));
    if(buffer == NULL){
        throw std::runtime_error("Not enough memory");
    }
    
    m_file.read(reinterpret_cast<char*>(buffer),n*dimentions*sizeof(float));
    
    for (size_t i = 0; i < n; i++)
    {
        std::shared_ptr<HDVector> vector_floats = std::make_shared<HDVector>(dimentions);
        std::copy_n(buffer + i*dimentions,dimentions,vector_floats->getDataPointer());
        vec->push_back(vector_floats);
    }
    
    free(buffer);
    return vec;    
}


