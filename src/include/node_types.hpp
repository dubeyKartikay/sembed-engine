#ifndef NODE_TYPES
#define NODE_TYPES

#include <cstdint>
#include <optional>
#include <vector>

using NodeId = uint64_t;
using NodeList = std::vector<NodeId>;
using OptionalNodeId = std::optional<NodeId>;

#endif
