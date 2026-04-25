#include "batch_stochastic_kmeans.hpp"

#include "dataset.hpp"
#include "node_types.hpp"
#include "utils.hpp"
#include "vector_view.hpp"

#include <cstdint>
#include <vector>

std::vector<Cluster> clusterizeData(DataSet &dataSet, uint64_t k,
                                    uint64_t iterations) {
  std::vector<Cluster> clusters;
  clusters.reserve(static_cast<size_t>(k));
  const NodeList initialCenters = generateRandomNumbers(k, dataSet.getN());
  for (uint64_t clusterId = 0; clusterId < initialCenters.size(); ++clusterId) {
    const NodeId center = initialCenters[static_cast<size_t>(clusterId)];
    Cluster cluster;
    Point centerPoint = {clusterId, dataSet.getRecordViewByIndex(center)};
    cluster.center = centerPoint;
    clusters.push_back(cluster);
  }

  if (clusters.empty()) {
    return clusters;
  }

  auto assignPoints = [&dataSet, &clusters]() {
    for (uint64_t i = 0; i < clusters.size(); ++i) {
      clusters[i].points.clear();
    }
    for (uint64_t i = 0; i < dataSet.getN(); ++i) {
      const RecordView record = dataSet.getRecordViewByIndex(i);
      const uint64_t closestCluster = getClosestCluster({0, record}, clusters);
      clusters[closestCluster].points.push_back({closestCluster, record});
    }
  };

  assignPoints();
  for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
    for (uint64_t i = 0; i < clusters.size(); i++) {
      clusters[i].center = newCenter(clusters[i]);
    }
    assignPoints();
  }

  return clusters;
}

uint64_t getClosestCluster(const Point &point,
                           const std::vector<Cluster> &clusters) {
  if (clusters.empty()) {
    return {};
  }
  uint64_t closestClusterId = 0;
  float minDistance = point.distanceSquared(clusters[0].center);
  for (uint64_t i = 1; i < clusters.size(); i++) {
    float distance = point.distanceSquared(clusters[i].center);
    if (distance < minDistance) {
      minDistance = distance;
      closestClusterId = i;
    }
  }
  return closestClusterId;
}

Point getClosestPoint(const Point &point,
                      const std::vector<Point> &otherPoints) {
  if (otherPoints.empty()) {
    return {};
  }
  Point closestPoint = otherPoints[0];
  float minDistance = point.distanceSquared(closestPoint);
  for (uint64_t i = 1; i < otherPoints.size(); i++) {
    float distance = point.distanceSquared(otherPoints[i]);
    if (distance < minDistance) {
      minDistance = distance;
      closestPoint = otherPoints[i];
    }
  }
  return closestPoint;
}

Point newCenter(const Cluster &cluster) {
  if (cluster.points.empty()) {
    return cluster.center;
  }

  const uint64_t dimensions = cluster.points[0].record.values.dimensions();
  std::vector<double> sums(static_cast<size_t>(dimensions), 0.0);

  for (const Point &point : cluster.points) {
    for (uint64_t dim = 0; dim < dimensions; ++dim) {
      sums[static_cast<size_t>(dim)] +=
          static_cast<double>(point.record.values[dim]);
    }
  }

  std::vector<float> centroid(static_cast<size_t>(dimensions), 0.0f);
  const double scale = 1.0 / static_cast<double>(cluster.points.size());
  for (uint64_t dim = 0; dim < dimensions; ++dim) {
    centroid[static_cast<size_t>(dim)] =
        static_cast<float>(sums[static_cast<size_t>(dim)] * scale);
  }

  Point centroidPoint = cluster.center;
  centroidPoint.record.values = FloatVectorView(centroid.data(), dimensions);
  return getClosestPoint(centroidPoint, cluster.points);
}
