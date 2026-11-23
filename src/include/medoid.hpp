#pragma once

#include "dataset.hpp"
#include "node_types.hpp"

namespace sembed_internal {

OptionalNodeId computeMedoidCpu(const DataSet& dataSet);
OptionalNodeId computeMedoid(const DataSet& dataSet);

#ifdef SEMBED_ENABLE_METAL
OptionalNodeId computeMedoidMetal(const DataSet& dataSet);
#endif

}  // namespace sembed_internal
