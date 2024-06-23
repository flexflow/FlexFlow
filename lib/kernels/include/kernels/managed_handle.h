#ifndef _FLEXFLOW_KERNELS_MANAGED_HANDLE_H
#define _FLEXFLOW_KERNELS_MANAGED_HANDLE_H

#include "kernels/ff_handle.h"

namespace FlexFlow {

struct ManagedHandle {
  PerDeviceFFHandle handle;

  ManagedHandle();

  ManagedHandle(ManagedHandle const &) = delete;
  ManagedHandle(ManagedHandle &&) = delete;

  ~ManagedHandle();
};

ManagedHandle get_managed_handle();

} // namespace FlexFlow

#endif
