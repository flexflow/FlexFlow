#ifndef _FLEXFLOW_KERNELS_MANAGED_FF_STREAM_H
#define _FLEXFLOW_KERNELS_MANAGED_FF_STREAM_H

#include "device.h"

namespace FlexFlow {

struct ManagedFFStream {
  ffStream_t stream;

  ManagedFFStream();

  ManagedFFStream(ManagedFFStream const &) = delete;
  ManagedFFStream(ManagedFFStream &&) = delete;

  ~ManagedFFStream();
};

} // namespace FlexFlow

#endif
