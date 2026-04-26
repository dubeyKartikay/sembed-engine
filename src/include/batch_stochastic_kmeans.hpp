#ifndef KMEANS
#define KMEANS

#include <cstdint>
#include <vector>

#include "dataset.hpp"
#include "node_types.hpp"
#include "vector_view.hpp"

struct Point {
  NodeId clusterId = 0;
  RecordView record;

  float distanceSquared(const Point &other) const {
    return squaredDistance(record.values, other.record.values);
  }

  float distanceSquared(const RecordView &other) const {
    return squaredDistance(record.values, other.values);
  }
};

struct Cluster {
  Point center;
  std::vector<Point> points;
};

std::vector<Cluster> clusterizeData(DataSet &vectorDataSet, uint64_t k = 40,
                                    uint64_t iterations = 100);
uint64_t getClosestCluster(const Point &point,
                           const std::vector<Cluster> &clusters);
Point getClosestPoint(const Point &point,
                      const std::vector<Point> &otherPoints);

Point newCenter(const Cluster &cluster);

#endif  // KMEANS
