#include <sembed/sembed.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dataset.hpp"
#include "vamana.hpp"
#include "vector_view.hpp"

namespace sembed {

struct Index::Impl {
  std::unique_ptr<Vamana> vamana;
  IndexConfig config;
};

namespace {

void validateIndexConfig(const IndexConfig& config) {
  if (config.degreeThreshold == 0) {
    throw std::invalid_argument("degree threshold must be positive");
  }
  if (config.searchListSize == 0) {
    throw std::invalid_argument("index search list size must be positive");
  }
  if (!(config.distanceThreshold > 0.0F) ||
      !std::isfinite(config.distanceThreshold)) {
    throw std::invalid_argument("distance threshold must be finite and positive");
  }
}

std::unique_ptr<Vamana> makeVamana(std::unique_ptr<DataSet> dataset,
                                   const IndexConfig& config) {
  validateIndexConfig(config);
  return std::make_unique<Vamana>(
      std::move(dataset), config.degreeThreshold, config.distanceThreshold,
      config.searchListSize);
}

}  // namespace

Index::Index(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Index::Index(Index&&) noexcept = default;
Index& Index::operator=(Index&&) noexcept = default;
Index::~Index() = default;

Index Index::load(const std::filesystem::path& path) {
  auto impl = std::make_unique<Impl>();
  impl->vamana = std::make_unique<Vamana>(path);
  impl->config = {impl->vamana->getDegreeThreshold(),
                  impl->vamana->getSearchListSize(),
                  impl->vamana->getDistanceThreshold()};
  return Index(std::move(impl));
}

void Index::save(const std::filesystem::path& path) const {
  if (!impl_ || !impl_->vamana) {
    throw std::logic_error("index has been moved from");
  }
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  impl_->vamana->saveIndex(path);
}

uint64_t Index::size() const {
  if (!impl_ || !impl_->vamana) {
    throw std::logic_error("index has been moved from");
  }
  return impl_->vamana->getNodeCount();
}

uint64_t Index::dimensions() const {
  if (!impl_ || !impl_->vamana) {
    throw std::logic_error("index has been moved from");
  }
  return impl_->vamana->getDimensions();
}

const IndexConfig& Index::config() const {
  if (!impl_ || !impl_->vamana) {
    throw std::logic_error("index has been moved from");
  }
  return impl_->config;
}

Index buildIndex(const std::filesystem::path& datasetPath,
                 const IndexConfig& config) {
  auto dataset = std::make_unique<FlatDataSet>(datasetPath);
  auto impl = std::make_unique<Index::Impl>();
  impl->vamana = makeVamana(std::move(dataset), config);
  impl->config = config;
  return Index(std::move(impl));
}

Index buildIndex(DatasetView dataset, const IndexConfig& config) {
  if (dataset.dimensions == 0) {
    throw std::invalid_argument("dataset dimensions must be positive");
  }
  if (dataset.size != 0 && dataset.vectors == nullptr) {
    throw std::invalid_argument("dataset vector data must not be null");
  }
  if (dataset.dimensions != 0 &&
      dataset.size > std::numeric_limits<uint64_t>::max() /
                         dataset.dimensions) {
    throw std::overflow_error("dataset shape is too large");
  }

  auto owned =
      std::make_unique<FlatDataSet>(dataset.dimensions, dataset.size);
  for (uint64_t node = 0; node < dataset.size; ++node) {
    const int64_t recordId =
        dataset.recordIds == nullptr
            ? static_cast<int64_t>(node)
            : dataset.recordIds[static_cast<size_t>(node)];
    const float* values =
        dataset.vectors + static_cast<size_t>(node * dataset.dimensions);
    owned->setVectorByIndex(node, recordId, values, dataset.dimensions);
  }
  auto impl = std::make_unique<Index::Impl>();
  impl->vamana = makeVamana(std::move(owned), config);
  impl->config = config;
  return Index(std::move(impl));
}

std::vector<Neighbor> query(const Index& index, FloatVectorView queryVector,
                            const QueryConfig& config) {
  if (!index.impl_ || !index.impl_->vamana) {
    throw std::logic_error("index has been moved from");
  }
  if (queryVector.dimensions() != index.dimensions()) {
    throw std::invalid_argument("query dimensions must match the index");
  }
  if (!queryVector.empty() && queryVector.data() == nullptr) {
    throw std::invalid_argument("query vector data must not be null");
  }
  const uint64_t searchListSize =
      config.searchListSize == 0 ? index.config().searchListSize
                                 : config.searchListSize;

  const SearchResults result = index.impl_->vamana->greedySearch(
      ::FloatVectorView(queryVector.data(), queryVector.dimensions()), config.k,
      searchListSize);

  std::vector<Neighbor> neighbors;
  neighbors.reserve(static_cast<size_t>(result.approximateNN.getSize()));
  for (uint64_t i = 0; i < result.approximateNN.getSize(); ++i) {
    const Neighbour& neighbor = result.approximateNN[i];
    const RecordView record =
        index.impl_->vamana->getRecordViewByIndex(neighbor.node);
    neighbors.push_back(
        {neighbor.node, record.recordId, std::sqrt(neighbor.distance)});
  }
  return neighbors;
}

}  // namespace sembed
