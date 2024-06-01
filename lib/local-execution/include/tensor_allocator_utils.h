#ifndef _FLEXFLOW_LOCAL_EXECUTION_ALLOCATOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_ALLOCATOR_H

#include "kernels/allocation.h"
#include "pcg/tensor.h"
#include <unordered_set>

namespace FlexFlow {

GenericTensorAccessorW allocate_tensor(Allocator allocator, Tensor tensor) {
  void* ptr = allocator.allocate(tensor.get_volume());
  return {tensor.data_type, tensor.get_shape(), ptr};
}

} // namespace FlexFlow

#endif
