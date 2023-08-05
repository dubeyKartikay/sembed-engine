#include <fstream>
#include "dataset.hpp"
#ifndef KMEANS
#define KMEANS
void clusterize_data(DataSet &vector_dataset, int k = 40,int M = 20,int iterations = 100);
#endif