#include <vector>
#include  <string>
#include <stdexcept>
#include <iostream>
#include "batch_stocastic_kmeans.hpp"
#include "dataset.hpp"

// todo : abstract away reading from binary file in a separate class;
std::vector<std::vector<float>*> * load_from_binary(std::string path){

    InMemoryDataSet vector_binary (path);
    clusterize_data(vector_binary);

    // std::vector<std::vector<float>*> *r_vector_data = new std::vector<std::vector<float>*>;
    // r_vector_data->reserve(n);
    // for (int i = 0; i < n; i++)
    // {
    //     std::vector<float> *vec_floats = new std::vector<float>(dimentions,0);
    //     vec_floats->reserve(dimentions);
    //     vector_binary.read(reinterpret_cast<char*>(vec_floats->data()),dimentions*sizeof(float));
    //     r_vector_data->push_back(vec_floats);
    // }

    // return r_vector_data;
    
    

}
