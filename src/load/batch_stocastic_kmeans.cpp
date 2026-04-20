#include <algorithm>
#include <fstream>
#include <cstdint>
#include<vector>
#include"dataset.hpp"
#include "utils.hpp"
void clusterize_data(DataSet &vector_binary, uint64_t k, uint64_t M,
                     uint64_t iterations){
    (void)M;
    (void)iterations;
    std::vector<HDVector> centers;
    // selecting random centroids    
    centers.reserve(static_cast<size_t>(k));
    for(int64_t center : generateRandomNumbers(k, vector_binary.getN())){
        centers.push_back(*vector_binary.getRecordViewByIndex(center).vector);
    }

    




}
