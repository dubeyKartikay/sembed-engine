#include "node_types.hpp"
#include "vamana.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
class DiskAnn {
private:
  Vamana m_vamana;

public:
  static void indexFromRaw(std::filesystem::path rawVectorBin, size_t k,
                           size_t l);
  NodeList search(NodeId queryNode, uint64_t k);
};
