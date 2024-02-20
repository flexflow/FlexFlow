#ifndef _FLEXFLOW_RUNTIME_SRC_LOCAL_ALLOCATOR_H
#define _FLEXFLOW_RUNTIME_SRC_LOCAL_ALLOCATOR_H

#include "kernels/allocation.h"
#include "runtime/src/tensor.h"
#include <unordered_map>

namespace FlexFlow {

struct LocalAllocator : public IAllocator {
  LocalAllocator(size_t);
  ~LocalAllocator() override;

  void *allocate(Tensor) override;
  void deallocate(void *) override;
  size_t get_ptr_memory_size(void *);

private:
  size_t total_memory_size;
  size_t allocated_memory_size;
  std::unordered_map<void *, size_t> ptr_memory_size_mapping;
};

Allocator get_local_memory_allocator(size_t);

} // namespace FlexFlow

#endif