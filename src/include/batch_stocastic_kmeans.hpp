#include <cstdint>
#include <fstream>
#include "dataset.hpp"
#include "node_types.hpp"
#ifndef KMEANS
#define KMEANS
struct Point{
  NodeId ClusterId;
  RecordView Record;
  float distance(const Point & other) const {
    return HDVector::distance(*Record.vector, *other.Record.vector);
  }

  float distance(const RecordView & other) const {
    return HDVector::distance(*Record.vector, *other.vector);
  }
};
struct Cluster{
  Point center;
  std::vector<Point> Points;
};
std::vector<Cluster> clusterize_data(DataSet &vector_dataset, uint64_t k = 40, uint64_t iterations = 100);
uint64_t getClosestCluster(const Point & point, const std::vector<Cluster> & clusters);
Point getClosestPoint(const Point & point, const std::vector<Point> & otherPoints);

Point newCenter(const Cluster & cluster);
#endif
