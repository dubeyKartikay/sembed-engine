#include <cstdint>
#include <fstream>
#include "dataset.hpp"
#ifndef KMEANS
#define KMEANS
void clusterize_data(DataSet &vector_dataset, uint64_t k = 40,
                     uint64_t M = 20, uint64_t iterations = 100);
#endif
