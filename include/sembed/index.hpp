#ifndef SEMBED_INDEX_HPP
#define SEMBED_INDEX_HPP

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <sembed/dataset_view.hpp>

namespace sembed {

class FloatVectorView;
struct Neighbor;
struct QueryConfig;

struct IndexConfig {
  uint64_t degreeThreshold = 64;
  uint64_t searchListSize = 100;
  float distanceThreshold = 1.2F;
};

class Index {
 public:
  Index(Index&&) noexcept;
  Index& operator=(Index&&) noexcept;
  ~Index();

  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;

  static Index load(const std::filesystem::path& path);
  void save(const std::filesystem::path& path) const;

  uint64_t size() const;
  uint64_t dimensions() const;
  const IndexConfig& config() const;

 private:
  struct Impl;
  explicit Index(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend Index buildIndex(const std::filesystem::path&, const IndexConfig&);
  friend Index buildIndex(DatasetView, const IndexConfig&);
  friend std::vector<Neighbor> query(const Index&, FloatVectorView,
                                     const QueryConfig&);
};

Index buildIndex(const std::filesystem::path& datasetPath,
                 const IndexConfig& config = {});
Index buildIndex(DatasetView dataset, const IndexConfig& config = {});

}  // namespace sembed

#endif  // SEMBED_INDEX_HPP
