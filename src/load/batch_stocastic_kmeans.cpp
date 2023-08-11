#include <algorithm>
#include <fstream>
#include<vector>
#include"dataset.hpp"
#include "utils.hpp"
void clusterize_data(DataSet &vector_binary, int k = 40,int M = 32,int iterations = 10){
    std::vector<HDVector> centers;
    // selecting random centroids    
    centers.reserve(k);
    for(int center : generateRandomNumbers(k,vector_binary.getN())){
        centers.push_back(*vector_binary.getHDVecByIndex(center));
    }

    




}
