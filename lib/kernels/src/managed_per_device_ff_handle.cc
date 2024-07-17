#include "kernels/managed_per_device_ff_handle.h"
#include "device.h"

namespace FlexFlow {

ManagedPerDeviceFFHandle::ManagedPerDeviceFFHandle() {
  handle = new PerDeviceFFHandle;
  handle->workSpaceSize = 1024 * 1024;
  handle->allowTensorOpMathConversion = true;

  checkCUDNN(cudnnCreate(&handle->dnn));
  checkCUBLAS(cublasCreate(&handle->blas));
  checkCUDA(cudaMalloc(&handle->workSpace, handle->workSpaceSize));
}

ManagedPerDeviceFFHandle::ManagedPerDeviceFFHandle(
    ManagedPerDeviceFFHandle &&other) noexcept
    : handle(std::exchange(other.handle, nullptr)) {}

ManagedPerDeviceFFHandle &ManagedPerDeviceFFHandle::operator=(
    ManagedPerDeviceFFHandle &&other) noexcept {
  std::swap(this->handle, other.handle);
  return *this;
}

ManagedPerDeviceFFHandle::~ManagedPerDeviceFFHandle() {
  if (handle != nullptr) {
    checkCUDNN(cudnnDestroy(handle->dnn));
    checkCUBLAS(cublasDestroy(handle->blas));
    checkCUDA(cudaFree(handle->workSpace));
    delete handle;
  }
}

PerDeviceFFHandle const &ManagedPerDeviceFFHandle::raw_handle() const {
  return *handle;
}

} // namespace FlexFlow
