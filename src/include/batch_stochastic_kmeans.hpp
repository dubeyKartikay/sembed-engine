#ifndef KMEANS
#define KMEANS

#include <cstdint>
#include <vector>

#include "dataset.hpp"
#include "node_types.hpp"

struct Point {
  NodeId clusterId;
  RecordView record;

  float distance(const Point &other) const {
    return HDVector::distance(*record.vector, *other.record.vector);
  }

  float distance(const RecordView &other) const {
    return HDVector::distance(*record.vector, *other.vector);
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
