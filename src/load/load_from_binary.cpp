#include <vector>
#include<string>
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include<iostream>
#include<iterator>
namespace fs = std::filesystem;
bool isValidPath(const std::string& path) {
    return fs::exists(path);
}

std::vector<std::vector<float>*> * load_from_binary(std::string path){
    if(!isValidPath(path)){
        throw std::invalid_argument("Path provided to load_binary does not exist");
    }

    std::ifstream vector_binary;
    vector_binary.open(path,std::ios::binary);
    if (!vector_binary.is_open())
    {
        throw std::runtime_error("Unable to open the file provided to load_binary");
    }
    long long int n;
    long long int dimentions;

    vector_binary.read((char *)&n,sizeof(n));
    vector_binary.read((char *)&dimentions,sizeof(dimentions));
    // vector_binary.seekg(0*(dimentions+1)*sizeof(float),std::ios_base::cur);
    std::vector<std::vector<float>*> *r_vector_data = new std::vector<std::vector<float>*>;
    r_vector_data->reserve(n);
    for (int i = 0; i < n; i++)
    {
        std::vector<float> *vec_floats = new std::vector<float>(dimentions,0);
        vec_floats->reserve(dimentions);
        vector_binary.read(reinterpret_cast<char*>(vec_floats->data()),dimentions*sizeof(float));
        r_vector_data->push_back(vec_floats);
    }

    return r_vector_data;
    
    

}
