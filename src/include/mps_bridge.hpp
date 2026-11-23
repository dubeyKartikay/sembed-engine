#pragma once

#include <cstdint>

namespace sembed_internal {

bool encodeMpsCentroid(void* mtlDevice,
                       void* mtlCommandBuffer,
                       void* vectorsBuffer,
                       void* onesBuffer,
                       void* centroidBuffer,
                       uint64_t rows,
                       uint64_t columns);

}  // namespace sembed_internal
