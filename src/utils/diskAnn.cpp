#include "DiskAnn.hpp"
#include "armadillo"
#include "dataset.hpp"
#include <algorithm>
#include <cstddef>
#include <mlpack.hpp>
#include <vector>

std::vector<size_t> getClosestClusters(arma::Col<float> vector, arma::mat centroids) {
  std::vector<size_t> result(centroids.n_cols);
  std::vector<double> distances(centroids.n_cols);
  for (size_t i = 0; i < centroids.n_cols; i++) {
    double dist = 0.0;
    auto centroid = centroids.col(i);
    for (size_t j = 0; j < vector.n_cols; j++) {
      dist += (double)(vector[j] - centroid[j]) * (double)(vector[j] - centroid[j]);
    }
    distances[i] = dist;
    result[i] = i;
  }

  std::sort(result.begin(), result.end(), [&distances](size_t a, size_t b) {
    return distances[a] < distances[b];
  });
  return result;
}

void DiskAnn::indexFromRaw(std::filesystem::path rawVectorBin,size_t k, size_t l) {
  auto dataset = FlatDataSet(rawVectorBin);
  arma::Row<size_t> assignments;
  arma::mat centroids;
  mlpack::KMeans<
        mlpack::EuclideanDistance,
        mlpack::RandomPartition,
        mlpack::MaxVarianceNewCluster,
        mlpack::NaiveKMeans,
        arma::fmat
    > kmeans;
  arma::fmat data(dataset.data(), dataset.getDimensions(), dataset.getN(),
                  false, true);
  kmeans.Cluster(data, k, assignments, centroids,false,false);
  std::vector<FlatDataSet> clusters(k); 
  for (size_t i = 0; i < dataset.getN(); i++) {
    auto closestClusters = getClosestClusters(data.col(i), centroids);
    for (size_t j = 0; j < l; j++) {
      auto vectorView = dataset.getRecordViewByIndex(i);
      clusters[closestClusters[j]].addVector(vectorView.recordId,data.colptr(i), dataset.getDimensions());
    }
  }

}


