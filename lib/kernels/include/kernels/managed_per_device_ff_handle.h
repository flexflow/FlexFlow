#ifndef _FLEXFLOW_KERNELS_MANAGED_HANDLE_H
#define _FLEXFLOW_KERNELS_MANAGED_HANDLE_H

#include "kernels/ff_handle.h"

namespace FlexFlow {

struct ManagedPerDeviceFFHandle {
  PerDeviceFFHandle handle;

  ManagedPerDeviceFFHandle();

  ~ManagedPerDeviceFFHandle();

  ManagedPerDeviceFFHandle(ManagedPerDeviceFFHandle &&other) noexcept
      : handle(std::move(other.handle)) {}

  ManagedPerDeviceFFHandle &
      operator=(ManagedPerDeviceFFHandle &&other) noexcept {
    if (this != &other) {
      handle = std::move(other.handle);
    }
    return *this;
  }

  ManagedPerDeviceFFHandle(ManagedPerDeviceFFHandle const &) = delete;

  ManagedPerDeviceFFHandle &
      operator=(ManagedPerDeviceFFHandle const &) = delete;
};

} // namespace FlexFlow

#endif
