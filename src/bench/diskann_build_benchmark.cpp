#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <sys/resource.h>

#include "DiskAnn.hpp"

namespace {

struct DatasetHeader {
  int64_t records = 0;
  int64_t storedDimensions = 0;
};

DatasetHeader readDatasetHeader(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open dataset");
  }

  DatasetHeader header;
  file.read(reinterpret_cast<char*>(&header.records), sizeof(header.records));
  file.read(reinterpret_cast<char*>(&header.storedDimensions),
            sizeof(header.storedDimensions));
  if (!file) {
    throw std::runtime_error("failed to read dataset header");
  }
  return header;
}

long peakResidentBytes() {
  rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0;
  }

#if defined(__APPLE__)
  return usage.ru_maxrss;
#else
  return usage.ru_maxrss * 1024L;
#endif
}

uint64_t parseUnsigned(const char* value, const std::string& name) {
  size_t parsed = 0;
  const std::string text(value);
  const auto result = std::stoull(text, &parsed);
  if (parsed != text.size()) {
    throw std::invalid_argument("invalid " + name + ": " + text);
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " <flat-dataset-bin> <clusters> "
              << "<assigned-clusters>\n";
    return 2;
  }

  try {
    const std::filesystem::path datasetPath(argv[1]);
    const uint64_t clusters = parseUnsigned(argv[2], "clusters");
    const uint64_t assignedClusters =
        parseUnsigned(argv[3], "assigned-clusters");
    const auto header = readDatasetHeader(datasetPath);

    const auto start = std::chrono::steady_clock::now();
    DiskAnn::indexFromRaw(datasetPath, clusters, assignedClusters);
    const auto end = std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsed = end - start;
    std::cout << "{\n"
              << "  \"dataset\": \"" << datasetPath.string() << "\",\n"
              << "  \"records\": " << header.records << ",\n"
              << "  \"dimensions\": " << (header.storedDimensions - 1)
              << ",\n"
              << "  \"clusters\": " << clusters << ",\n"
              << "  \"assigned_clusters\": " << assignedClusters << ",\n"
              << "  \"build_seconds\": " << elapsed.count() << ",\n"
              << "  \"peak_resident_bytes\": " << peakResidentBytes() << "\n"
              << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "diskann_build_benchmark: " << error.what() << '\n';
    return 1;
  }
}
