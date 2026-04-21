#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "graph.hpp"

namespace testutils {

inline std::string sanitizePathComponent(std::string value) {
  for (char &ch : value) {
    const bool is_alnum = (ch >= 'a' && ch <= 'z') ||
                          (ch >= 'A' && ch <= 'Z') ||
                          (ch >= '0' && ch <= '9');
    if (!is_alnum && ch != '_' && ch != '-') {
      ch = '_';
    }
  }
  return value;
}

inline std::filesystem::path fixtureDir() {
  const auto dir = std::filesystem::current_path() / "build" / "test-fixtures";
  std::filesystem::create_directories(dir);
  return dir;
}

inline std::filesystem::path namedFixturePath(const std::string &name) {
  return fixtureDir() / name;
}

inline std::filesystem::path uniqueFixturePath(const std::string &prefix,
                                               const std::string &tag,
                                               const std::string &extension = ".bin") {
  const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string suite =
      info ? sanitizePathComponent(info->test_suite_name()) : "unknown_suite";
  const std::string name =
      info ? sanitizePathComponent(info->name()) : "unknown_test";
  return fixtureDir() /
         (prefix + "_" + suite + "_" + name + "_" + tag + extension);
}

struct ScopedPathCleanup {
  explicit ScopedPathCleanup(std::filesystem::path path)
      : path(std::move(path)) {}

  ~ScopedPathCleanup() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  std::filesystem::path path;
};

inline std::filesystem::path writeGraphFile(
    const std::filesystem::path &path, uint64_t nodes,
    uint64_t degree_threshold, uint64_t medoid,
    const std::vector<NodeList> &adjacency);

inline std::filesystem::path writeDatasetFile(
    const std::filesystem::path &path, int64_t n, int64_t stored_dimensions,
    const std::vector<std::vector<float>> &rows) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open dataset fixture for writing");
  }
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(&stored_dimensions),
            sizeof(stored_dimensions));
  for (const auto &row : rows) {
    if (static_cast<int64_t>(row.size()) != stored_dimensions) {
      throw std::invalid_argument(
          "dataset fixture row width does not match header");
    }
    const int64_t record_id = static_cast<int64_t>(row.front());
    out.write(reinterpret_cast<const char *>(&record_id), sizeof(record_id));
    out.write(reinterpret_cast<const char *>(row.data() + 1),
              static_cast<std::streamsize>((row.size() - 1) * sizeof(float)));
  }
  return path;
}

inline std::filesystem::path writeGraphFile(
    const std::filesystem::path &path, uint64_t nodes, uint64_t degree_threshold,
    const std::vector<NodeList> &adjacency) {
  const uint64_t no_medoid = std::numeric_limits<uint64_t>::max();
  return writeGraphFile(path, nodes, degree_threshold, no_medoid, adjacency);
}

inline std::filesystem::path writeGraphFile(
    const std::filesystem::path &path, uint64_t nodes, uint64_t degree_threshold,
    uint64_t medoid, const std::vector<NodeList> &adjacency) {
  if (adjacency.size() != static_cast<size_t>(nodes)) {
    throw std::invalid_argument(
        "graph fixture adjacency size does not match node count");
  }
  Graph graph(nodes, degree_threshold);
  for (uint64_t node = 0; node < nodes; ++node) {
    const auto &neighbours = adjacency[static_cast<size_t>(node)];
    if (static_cast<uint64_t>(neighbours.size()) > degree_threshold) {
      throw std::invalid_argument(
          "graph fixture adjacency exceeds degree threshold");
    }
    graph.setOutNeighbors(node, neighbours);
  }

  graph.save(path);

  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  if (!file.is_open()) {
    throw std::runtime_error("failed to reopen graph fixture for patching");
  }
  file.seekp(static_cast<std::streamoff>(sizeof(uint64_t) * 2), std::ios::beg);
  file.write(reinterpret_cast<const char *>(&medoid), sizeof(medoid));
  if (!file) {
    throw std::runtime_error("failed to patch graph fixture medoid");
  }
  return path;
}

inline std::vector<NodeList> makeCircularAdjacency(
    uint64_t node_count, uint64_t degree_threshold) {
  std::vector<NodeList> adjacency(static_cast<size_t>(node_count));
  for (uint64_t node = 0; node < node_count; ++node) {
    adjacency[node].reserve(static_cast<size_t>(degree_threshold));
    for (uint64_t offset = 1; offset <= degree_threshold; ++offset) {
      adjacency[node].push_back((node + offset) % node_count);
    }
  }
  return adjacency;
}

inline std::filesystem::path embeddingFixturePath(const std::string &name) {
  return std::filesystem::path("../build") / name;
}

inline constexpr uint64_t kGloveFixtureRows = 256;
inline constexpr uint64_t kGloveFixtureDimensions = 50;

inline const std::vector<float> &firstGloveVector() {
  static const std::vector<float> data = {
      0.418f,     0.24968f,  -0.41242f, 0.1217f,    0.34527f,  -0.044457f,
      -0.49688f,  -0.17862f, -0.00066023f, -0.6566f, 0.27843f,  -0.14767f,
      -0.55677f,  0.14658f,  -0.0095095f, 0.011658f, 0.10204f,  -0.12792f,
      -0.8443f,   -0.12181f, -0.016801f, -0.33279f, -0.1552f,   -0.23131f,
      -0.19181f,  -1.8823f,  -0.76746f,  0.099051f, -0.42125f,  -0.19526f,
      4.0071f,    -0.18594f, -0.52287f,  -0.31681f, 0.00059213f, 0.0074449f,
      0.17778f,   -0.15897f, 0.012041f,  -0.054223f, -0.29871f, -0.15749f,
      -0.34758f,  -0.045637f, -0.44251f, 0.18785f,   0.0027849f, -0.18411f,
      -0.11514f,  -0.78581f};
  return data;
}

inline const std::vector<float> &lastGloveVector() {
  static const std::vector<float> data = {
      0.072617f, -0.51393f, 0.4728f,   -0.52202f, -0.35534f, 0.34629f,
      0.23211f,  0.23096f,  0.26694f,  0.41028f,  0.28031f,  0.14107f,
      -0.30212f, -0.21095f, -0.10875f, -0.33659f, -0.46313f, -0.40999f,
      0.32764f,  0.47401f,  -0.43449f, 0.19959f,  -0.55808f, -0.34077f,
      0.078477f, 0.62823f,  0.17161f,  -0.34454f, -0.2066f,  0.1323f,
      -1.8076f,  -0.38851f, 0.37654f,  -0.50422f, -0.012446f, 0.046182f,
      0.70028f,  -0.010573f, -0.83629f, -0.24698f, 0.6888f,   -0.17986f,
      -0.066569f, -0.48044f, -0.55946f, -0.27594f, 0.056072f, -0.18907f,
      -0.59021f, 0.55559f};
  return data;
}

}  // namespace testutils
