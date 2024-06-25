#include "kernels/managed_per_device_ff_handle.h"
#include "device.h"

namespace FlexFlow {
ManagedPerDeviceFFHandle::ManagedPerDeviceFFHandle() {
  handle.workSpaceSize = 1024 * 1024;
  handle.allowTensorOpMathConversion = true;

  checkCUDNN(cudnnCreate(&handle.dnn));
  checkCUBLAS(cublasCreate(&handle.blas));
  checkCUDA(cudaMalloc(&handle.workSpace, handle.workSpaceSize));
}

ManagedPerDeviceFFHandle::~ManagedPerDeviceFFHandle() {
  checkCUDNN(cudnnDestroy(handle.dnn));
  checkCUBLAS(cublasDestroy(handle.blas));
  checkCUDA(cudaFree(handle.workSpace));
}

} // namespace FlexFlow
