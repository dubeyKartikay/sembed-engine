#include "dataset.hpp"
#include <iostream>
InMemoryDataSet::InMemoryDataSet(fs::path path){
    FileDataSet * f = new FileDataSet(path);
    this->n =  f->getN();
    this->dimentions = f->getDimentions();
    this->getN();
    std::cout << this->n << " " << this->dimentions << "\n" ; 
    this->m_data = f->getNVectorsFromIndex(0,this->getN());
 /*    delete f; */
}

std::vector<float>* InMemoryDataSet::getVecByIndex(const int & index){
    std::vector<float> * f = new std::vector<float> (this->m_data->at(index)->begin(),this->m_data->at(index)->end());
    return f;
}
std::vector<std::vector<float>*>* InMemoryDataSet::getNVectorsFromIndex(const int  &index,const int & n){
        std::vector<std::vector<float>*>* vec = new  std::vector<std::vector<float>*>(this->m_data->begin()+index,this->m_data->begin()+index + n);
        return vec;
 }
