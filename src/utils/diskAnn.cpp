#include "DiskAnn.hpp"
#include "armadillo"
#include "dataset.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <vector>

std::vector<size_t> getClosestClusters(const arma::fvec& vector,
                                       const arma::fmat& centroids) {
  std::vector<size_t> result(centroids.n_cols);
  std::vector<double> distances(centroids.n_cols);
  for (size_t i = 0; i < centroids.n_cols; i++) {
    double dist = 0.0;
    auto centroid = centroids.col(i);
    for (size_t j = 0; j < vector.n_elem; j++) {
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

void DiskAnn::indexFromRaw(std::filesystem::path rawVectorBin, size_t k,
                           size_t l) {
  if (k == 0) {
    throw std::invalid_argument("cluster count must be positive");
  }
  if (l > k) {
    throw std::invalid_argument("assigned cluster count cannot exceed cluster count");
  }

  auto dataset = FlatDataSet(rawVectorBin);
  const auto recordCount = static_cast<long long>(dataset.getN());
  const auto dimensions = dataset.getDimensions();

  arma::fmat data(dataset.data(), dataset.getDimensions(), dataset.getN(),
                  false, true);

  arma::fmat centroids;
  if (!arma::kmeans(centroids, data, k, arma::random_subset, 100, false)) {
    throw std::runtime_error("k-means clustering failed");
  }

  std::vector<uint64_t> clusterSizes(k, 0);
  #pragma omp parallel
  {
    std::vector<uint64_t> localClusterSizes(k, 0);

    #pragma omp for schedule(static)
    for (long long i = 0; i < recordCount; i++) {
      auto closestClusters = getClosestClusters(data.col(i), centroids);
      for (size_t j = 0; j < l; j++) {
        localClusterSizes[closestClusters[j]]++;
      }
    }

    #pragma omp critical
    {
      for (size_t cluster = 0; cluster < k; cluster++) {
        clusterSizes[cluster] += localClusterSizes[cluster];
      }
    }
  }

  std::vector<FlatDataSet> clusters;
  clusters.reserve(k);
  for (size_t cluster = 0; cluster < k; cluster++) {
    clusters.emplace_back(dimensions, clusterSizes[cluster]);
  }

  std::vector<std::atomic<uint64_t>> nextClusterOffsets(k);
  for (size_t cluster = 0; cluster < k; cluster++) {
    nextClusterOffsets[cluster].store(0, std::memory_order_relaxed);
  }

  #pragma omp parallel for schedule(static)
  for (long long i = 0; i < recordCount; i++) {
    auto closestClusters = getClosestClusters(data.col(i), centroids);
    auto vectorView = dataset.getRecordViewByIndex(i);
    for (size_t j = 0; j < l; j++) {
      const size_t cluster = closestClusters[j];
      const uint64_t offset =
          nextClusterOffsets[cluster].fetch_add(1, std::memory_order_relaxed);
      clusters[cluster].setVectorByIndex(offset, vectorView.recordId,
                                         vectorView.values.data(), dimensions);
    }
  }

}
