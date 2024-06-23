#include "kernels/managed_handle.h"

namespace FlexFlow {
ManagedHandle::ManagedHandle() {
  handle.workSpaceSize = 1024 * 1024;
  handle.allowTensorOpMathConversion = true;

  cudnnCreate(&handle.dnn);
  cublasCreate(&handle.blas);
  checkCUDA(cudaMalloc(&handle.workSpace, handle.workSpaceSize));
}

ManagedHandle::~ManagedHandle() {
  cudnnDestroy(handle.dnn);
  cublasDestroy(handle.blas);
  checkCUDA(cudaFree(handle.workSpace));
}

ManagedHandle get_managed_handle() {
  return ManagedHandle();
}
} // namespace FlexFlow
