#include "kernels/managed_per_device_ff_handle.h"
#include "device.h"

namespace FlexFlow {

ManagedPerDeviceFFHandle::ManagedPerDeviceFFHandle() {
  this->handle = new PerDeviceFFHandle;
  this->handle->workSpaceSize = 1024 * 1024;
  this->handle->allowTensorOpMathConversion = true;

  checkCUDNN(cudnnCreate(&this->handle->dnn));
  checkCUBLAS(cublasCreate(&this->handle->blas));
  checkCUDA(cudaMalloc(&this->handle->workSpace, this->handle->workSpaceSize));
}

ManagedPerDeviceFFHandle::ManagedPerDeviceFFHandle(
    ManagedPerDeviceFFHandle &&other) noexcept
    : handle(std::exchange(other.handle, nullptr)) {}

ManagedPerDeviceFFHandle &ManagedPerDeviceFFHandle::operator=(
    ManagedPerDeviceFFHandle &&other) noexcept {
  if (this != &other) {
    if (this->handle != nullptr) {
      checkCUDNN(cudnnDestroy(this->handle->dnn));
      checkCUBLAS(cublasDestroy(this->handle->blas));
      checkCUDA(cudaFree(this->handle->workSpace));
      delete this->handle;
    }
    this->handle = std::exchange(other.handle, nullptr);
  }
  return *this;
}

ManagedPerDeviceFFHandle::~ManagedPerDeviceFFHandle() {
  if (this->handle != nullptr) {
    checkCUDNN(cudnnDestroy(this->handle->dnn));
    checkCUBLAS(cublasDestroy(this->handle->blas));
    checkCUDA(cudaFree(this->handle->workSpace));
    delete this->handle;
    this->handle = nullptr;
  }
}

PerDeviceFFHandle const &ManagedPerDeviceFFHandle::raw_handle() const {
  return *handle;
}

} // namespace FlexFlow
