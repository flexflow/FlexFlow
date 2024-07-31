#ifndef _FLEXFLOW_KERNELS_MANAGED_HANDLE_H
#define _FLEXFLOW_KERNELS_MANAGED_HANDLE_H

#include "kernels/ff_handle.h"

namespace FlexFlow {

struct ManagedPerDeviceFFHandle {
public:
  ManagedPerDeviceFFHandle();

  ManagedPerDeviceFFHandle(ManagedPerDeviceFFHandle const &) = delete;
  ManagedPerDeviceFFHandle &
      operator=(ManagedPerDeviceFFHandle const &) = delete;

  ManagedPerDeviceFFHandle(ManagedPerDeviceFFHandle &&other) noexcept;
  ManagedPerDeviceFFHandle &
      operator=(ManagedPerDeviceFFHandle &&other) noexcept;

  ~ManagedPerDeviceFFHandle();

  PerDeviceFFHandle const &raw_handle() const;

private:
  PerDeviceFFHandle *handle;
};

} // namespace FlexFlow

#endif
