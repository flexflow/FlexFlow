#ifndef _FLEXFLOW_KERNELS_MANAGED_HANDLE_H
#define _FLEXFLOW_KERNELS_MANAGED_HANDLE_H

#include "kernels/ff_handle.h"

namespace FlexFlow {

struct ManagedPerDeviceFFHandle {
  PerDeviceFFHandle handle;

  ManagedPerDeviceFFHandle();

  ~ManagedPerDeviceFFHandle();
};

} // namespace FlexFlow

#endif
