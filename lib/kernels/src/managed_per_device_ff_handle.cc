#include "kernels/managed_per_device_ff_handle.h"
#include "internal/device.h"
#include "kernels/nccl.h"

namespace FlexFlow {

ManagedPerDeviceFFHandle::ManagedPerDeviceFFHandle(
    int num_ranks,
    int my_rank,
    size_t workSpaceSize,
    bool allowTensorOpMathConversion) {
  this->handle = new PerDeviceFFHandle{};
  this->handle->workSpaceSize = workSpaceSize;
  this->handle->allowTensorOpMathConversion = allowTensorOpMathConversion;

  checkCUDNN(cudnnCreate(&this->handle->dnn));
  checkCUBLAS(cublasCreate(&this->handle->blas));
  checkCUDA(cudaMalloc(&this->handle->workSpace, this->handle->workSpaceSize));

#ifdef FF_USE_NCCL
  // A single-rank communicator can only ever perform no-op collectives, but
  // still costs ~500MiB of device memory, so skip creating it entirely.
  if (num_ranks == 1) {
    this->handle->ncclComm = nullptr;
  } else {
    ncclUniqueId ncclId;
    checkNCCL(ncclGetUniqueId(&ncclId));
    checkNCCL(ncclCommInitRank(
        &handle->ncclComm, num_ranks, ncclId, my_rank)); // todo generalize
  }
#endif
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
  if (this->handle != nullptr) {
    checkCUDNN(cudnnDestroy(this->handle->dnn));
    checkCUBLAS(cublasDestroy(this->handle->blas));
    checkCUDA(cudaFree(this->handle->workSpace));
#ifdef FF_USE_NCCL
    if (this->handle->ncclComm != nullptr) {
      checkNCCL(ncclCommDestroy(this->handle->ncclComm));
    }
#endif
    delete this->handle;
  }
}

PerDeviceFFHandle const &ManagedPerDeviceFFHandle::raw_handle() const {
  return *handle;
}

std::optional<ManagedPerDeviceFFHandle>
    create_local_handle_for_device_type(DeviceType device_type,
                                        size_t workSpaceSize,
                                        bool allowTensorOpMathConversion) {
  if (device_type == DeviceType::CPU) {
    return std::nullopt;
  } else {
    return initialize_single_gpu_handle(
        /*workSpaceSize=*/workSpaceSize,
        /*allowTensorOpMathConversion=*/allowTensorOpMathConversion);
  }
}

ManagedPerDeviceFFHandle
    initialize_single_gpu_handle(size_t workSpaceSize,
                                 bool allowTensorOpMathConversion) {
  return ManagedPerDeviceFFHandle{
      /*num_ranks=*/1,
      /*my_rank=*/0,
      /*workSpaceSize=*/workSpaceSize,
      /*allowTensorOpMathConversion=*/allowTensorOpMathConversion,
  };
}

ManagedPerDeviceFFHandle
    initialize_multi_gpu_handle(int num_ranks,
                                int my_rank,
                                size_t workSpaceSize,
                                bool allowTensorOpMathConversion) {
  return ManagedPerDeviceFFHandle{
      /*num_ranks=*/num_ranks,
      /*my_rank=*/my_rank,
      /*workSpaceSize=*/workSpaceSize,
      /*allowTensorOpMathConversion=*/allowTensorOpMathConversion,
  };
}

} // namespace FlexFlow
