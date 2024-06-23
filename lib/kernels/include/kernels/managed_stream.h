#ifndef _FLEXFLOW_KERNELS_MANAGED_STREAM_H
#define _FLEXFLOW_KERNELS_MANAGED_STREAM_H

#include "device.h"

namespace FlexFlow {

struct ManagedStream {
  ffStream_t stream;

  ManagedStream();

  ManagedStream(ManagedStream const &) = delete;
  ManagedStream(ManagedStream &&) = delete;

  ~ManagedStream();
};

ManagedStream get_managed_stream();

} // namespace FlexFlow

#endif
