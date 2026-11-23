#import "mps_bridge.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace sembed_internal {

bool encodeMpsCentroid(void* mtlDevice,
                       void* mtlCommandBuffer,
                       void* vectorsBuffer,
                       void* onesBuffer,
                       void* centroidBuffer,
                       uint64_t rows,
                       uint64_t columns) {
  if (!mtlDevice || !mtlCommandBuffer || !vectorsBuffer || !onesBuffer ||
      !centroidBuffer || rows == 0 || columns == 0) {
    return false;
  }

  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)mtlCommandBuffer;
    id<MTLBuffer> vecBuf = (__bridge id<MTLBuffer>)vectorsBuffer;
    id<MTLBuffer> oneBuf = (__bridge id<MTLBuffer>)onesBuffer;
    id<MTLBuffer> outBuf = (__bridge id<MTLBuffer>)centroidBuffer;

    const NSUInteger nRows = static_cast<NSUInteger>(rows);
    const NSUInteger nCols = static_cast<NSUInteger>(columns);
    const NSUInteger rowBytes = nCols * sizeof(float);

    MPSMatrixDescriptor* matrixDesc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:nRows
                                              columns:nCols
                                             rowBytes:rowBytes
                                             dataType:MPSDataTypeFloat32];
    MPSVectorDescriptor* onesDesc =
        [MPSVectorDescriptor vectorDescriptorWithLength:nRows
                                               dataType:MPSDataTypeFloat32];
    MPSVectorDescriptor* centroidDesc =
        [MPSVectorDescriptor vectorDescriptorWithLength:nCols
                                               dataType:MPSDataTypeFloat32];
    if (!matrixDesc || !onesDesc || !centroidDesc) {
      return false;
    }

    MPSMatrix* inputMatrix = [[MPSMatrix alloc] initWithBuffer:vecBuf
                                                    descriptor:matrixDesc];
    MPSVector* onesVector = [[MPSVector alloc] initWithBuffer:oneBuf
                                                   descriptor:onesDesc];
    MPSVector* centroidVector = [[MPSVector alloc] initWithBuffer:outBuf
                                                       descriptor:centroidDesc];
    if (!inputMatrix || !onesVector || !centroidVector) {
      return false;
    }

    // Computes centroid = (1 / rows) * X^T * 1_rows.
    // With transpose = YES, op(A) has nCols rows and nRows columns.
    MPSMatrixVectorMultiplication* matvec =
        [[MPSMatrixVectorMultiplication alloc]
            initWithDevice:device
                 transpose:YES
                      rows:nCols
                   columns:nRows
                     alpha:(1.0 / static_cast<double>(rows))
                      beta:0.0];
    if (!matvec) {
      return false;
    }

    [matvec encodeToCommandBuffer:commandBuffer
                      inputMatrix:inputMatrix
                      inputVector:onesVector
                     resultVector:centroidVector];
    return true;
  }
}

}  // namespace sembed_internal
