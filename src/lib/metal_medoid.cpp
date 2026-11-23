#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "medoid.hpp"
#include "mps_bridge.hpp"

namespace sembed_internal {
namespace {

struct MedoidParams {
  uint32_t rows;
  uint32_t dims;
};

struct PartialMin {
  float dist;
  uint32_t index;
};

constexpr const char* kMedoidArgminShader = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct MedoidParams {
    uint rows;
    uint dims;
};

struct PartialMin {
    float dist;
    uint index;
};

kernel void medoid_distance_argmin(
    device const float* vectors [[buffer(0)]],
    device const float* centroid [[buffer(1)]],
    constant MedoidParams& params [[buffer(2)]],
    device PartialMin* partials [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint gridSize [[threads_per_grid]],
    uint tgId [[threadgroup_position_in_grid]])
{
    float bestDist = INFINITY;
    uint bestIdx = 0xFFFFFFFFu;

    for (uint row = gid; row < params.rows; row += gridSize) {
        device const float* rowPtr =
            vectors + static_cast<ulong>(row) * static_cast<ulong>(params.dims);
        float dist = 0.0f;
        for (uint d = 0; d < params.dims; ++d) {
            float diff = rowPtr[d] - centroid[d];
            dist = fma(diff, diff, dist);
        }
        if (dist < bestDist || (dist == bestDist && row < bestIdx)) {
            bestDist = dist;
            bestIdx = row;
        }
    }

    threadgroup float sDist[256];
    threadgroup uint sIdx[256];
    sDist[tid] = bestDist;
    sIdx[tid] = bestIdx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float otherDist = sDist[tid + stride];
            uint otherIdx = sIdx[tid + stride];
            if (otherDist < sDist[tid] ||
                (otherDist == sDist[tid] && otherIdx < sIdx[tid])) {
                sDist[tid] = otherDist;
                sIdx[tid] = otherIdx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partials[tgId].dist = sDist[0];
        partials[tgId].index = sIdx[0];
    }
}
)METAL";

struct MetalMedoidContext {
  MTL::Device* device = nullptr;
  MTL::CommandQueue* queue = nullptr;
  MTL::ComputePipelineState* argminPipeline = nullptr;
  uint32_t threadgroupSize = 256;

  MetalMedoidContext() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
      pool->release();
      return;
    }

    queue = device->newCommandQueue();
    if (!queue) {
      device->release();
      device = nullptr;
      pool->release();
      return;
    }

    NS::Error* error = nullptr;
    NS::String* source =
        NS::String::string(kMedoidArgminShader, NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(source, nullptr, &error);
    if (!library) {
      queue->release();
      device->release();
      queue = nullptr;
      device = nullptr;
      pool->release();
      return;
    }

    NS::String* fnName =
        NS::String::string("medoid_distance_argmin", NS::UTF8StringEncoding);
    MTL::Function* fn = library->newFunction(fnName);
    if (fn) {
      argminPipeline = device->newComputePipelineState(fn, &error);
      fn->release();
    }
    library->release();

    if (!argminPipeline) {
      queue->release();
      device->release();
      queue = nullptr;
      device = nullptr;
      pool->release();
      return;
    }

    const uint64_t maxThreads =
        argminPipeline->maxTotalThreadsPerThreadgroup();
    uint32_t tg = 256;
    while (tg > maxThreads && tg > 1) {
      tg >>= 1;
    }
    threadgroupSize = tg;
    pool->release();
  }

  ~MetalMedoidContext() {
    if (argminPipeline) {
      argminPipeline->release();
    }
    if (queue) {
      queue->release();
    }
    if (device) {
      device->release();
    }
  }

  bool ready() const {
    return device != nullptr && queue != nullptr && argminPipeline != nullptr;
  }
};

const MetalMedoidContext& getContext() {
  static const MetalMedoidContext context;
  return context;
}

}  // namespace

OptionalNodeId computeMedoidMetal(const DataSet& dataSet) {
  const uint64_t n = dataSet.getN();
  if (n == 0) {
    return std::nullopt;
  }
  const uint64_t dims = dataSet.getDimensions();
  if (n == 1 || dims == 0) {
    return NodeId{0};
  }

  const float* raw = dataSet.data();
  const MetalMedoidContext& ctx = getContext();
  if (raw == nullptr || !ctx.ready() ||
      n > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
      dims > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return computeMedoidCpu(dataSet);
  }

  const uint64_t vectorFloats = n * dims;
  if (vectorFloats / dims != n ||
      vectorFloats > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
    return computeMedoidCpu(dataSet);
  }
  const uint64_t vectorBytes = vectorFloats * sizeof(float);
  const uint64_t onesBytes = n * sizeof(float);
  const uint64_t centroidBytes = dims * sizeof(float);
  const uint64_t maxBufLen = static_cast<uint64_t>(ctx.device->maxBufferLength());
  if (vectorBytes > maxBufLen || onesBytes > maxBufLen ||
      centroidBytes > maxBufLen) {
    return computeMedoidCpu(dataSet);
  }

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  MTL::Buffer* vectorsBuffer =
      ctx.device->newBuffer(raw, static_cast<NS::UInteger>(vectorBytes),
                            MTL::ResourceStorageModeShared);
  MTL::Buffer* onesBuffer =
      ctx.device->newBuffer(static_cast<NS::UInteger>(onesBytes),
                            MTL::ResourceStorageModeShared);
  MTL::Buffer* centroidBuffer =
      ctx.device->newBuffer(static_cast<NS::UInteger>(centroidBytes),
                            MTL::ResourceStorageModeShared);

  const uint32_t tgSize = ctx.threadgroupSize;
  const uint64_t numGroups =
      std::min<uint64_t>((n + tgSize - 1) / tgSize, 1024ULL);
  MTL::Buffer* partialsBuffer = ctx.device->newBuffer(
      static_cast<NS::UInteger>(numGroups * sizeof(PartialMin)),
      MTL::ResourceStorageModeShared);

  if (!vectorsBuffer || !onesBuffer || !centroidBuffer || !partialsBuffer) {
    if (partialsBuffer) partialsBuffer->release();
    if (centroidBuffer) centroidBuffer->release();
    if (onesBuffer) onesBuffer->release();
    if (vectorsBuffer) vectorsBuffer->release();
    pool->release();
    return computeMedoidCpu(dataSet);
  }

  float* onesPtr = static_cast<float*>(onesBuffer->contents());
  std::fill(onesPtr, onesPtr + n, 1.0F);

  MTL::CommandBuffer* commandBuffer = ctx.queue->commandBuffer();
  const bool mpsOk =
      commandBuffer != nullptr &&
      encodeMpsCentroid(ctx.device, commandBuffer, vectorsBuffer, onesBuffer,
                        centroidBuffer, n, dims);

  if (!mpsOk) {
    partialsBuffer->release();
    centroidBuffer->release();
    onesBuffer->release();
    vectorsBuffer->release();
    pool->release();
    return computeMedoidCpu(dataSet);
  }

  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  const MedoidParams params{static_cast<uint32_t>(n),
                            static_cast<uint32_t>(dims)};
  encoder->setComputePipelineState(ctx.argminPipeline);
  encoder->setBuffer(vectorsBuffer, 0, 0);
  encoder->setBuffer(centroidBuffer, 0, 1);
  encoder->setBytes(&params, sizeof(params), 2);
  encoder->setBuffer(partialsBuffer, 0, 3);
  encoder->dispatchThreadgroups(
      MTL::Size::Make(static_cast<NS::UInteger>(numGroups), 1, 1),
      MTL::Size::Make(static_cast<NS::UInteger>(tgSize), 1, 1));
  encoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  OptionalNodeId result = std::nullopt;
  if (commandBuffer->status() == MTL::CommandBufferStatusCompleted) {
    const PartialMin* partials =
        static_cast<const PartialMin*>(partialsBuffer->contents());
    float bestDist = std::numeric_limits<float>::infinity();
    uint32_t bestIdx = std::numeric_limits<uint32_t>::max();
    for (uint64_t g = 0; g < numGroups; ++g) {
      if (partials[g].index == std::numeric_limits<uint32_t>::max()) {
        continue;
      }
      if (partials[g].dist < bestDist ||
          (partials[g].dist == bestDist && partials[g].index < bestIdx)) {
        bestDist = partials[g].dist;
        bestIdx = partials[g].index;
      }
    }
    if (bestIdx < n) {
      result = static_cast<NodeId>(bestIdx);
    }
  }

  partialsBuffer->release();
  centroidBuffer->release();
  onesBuffer->release();
  vectorsBuffer->release();
  pool->release();

  if (!result) {
    return computeMedoidCpu(dataSet);
  }
  return result;
}

}  // namespace sembed_internal
