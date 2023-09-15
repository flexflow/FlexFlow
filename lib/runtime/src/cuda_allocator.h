#ifndef _FLEXFLOW_RUNTIME_CUDA_ALLOCATOR_H
#define _FLEXFLOW_RUNTIME_CUDA_ALLOCATOR_H

#include "kernels/allocation.h"
#include <cuda_runtime.h>

namespace FlexFlow {

struct CudaAllocator : public IAllocator {
  ~CudaAllocator() override;

  void *allocate(size_t) override;
  void deallocate(void *) override;
};

} // namespace FlexFlow

#endif
