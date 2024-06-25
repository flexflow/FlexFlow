#ifndef _FLEXFLOW_KERNELS_MANAGED_FF_STREAM_H
#define _FLEXFLOW_KERNELS_MANAGED_FF_STREAM_H

#include "device.h"

namespace FlexFlow {

struct ManagedFFStream {
  ffStream_t stream;

  ManagedFFStream();

  ~ManagedFFStream();

  ManagedFFStream(ManagedFFStream &&other) noexcept
      : stream(std::exchange(other.stream, nullptr)) {}

  ManagedFFStream &operator=(ManagedFFStream &&other) noexcept {
    if (this != &other) {
      checkCUDA(cudaStreamDestroy(stream));
      stream = std::exchange(other.stream, nullptr);
    }
    return *this;
  }

  ManagedFFStream(ManagedFFStream const &) = delete;

  ManagedFFStream &operator=(ManagedFFStream const &) = delete;
};

} // namespace FlexFlow

#endif
