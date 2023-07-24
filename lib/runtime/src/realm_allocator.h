#ifndef _FLEXFLOW_RUNTIME_REALM_ALLOCATOR_H
#define _FLEXFLOW_RUNTIME_REALM_ALLOCATOR_H

#include "kernels/allocation.h"
#include "legion.h"
#include "utils/stack_vector.h"
#include <memory>

#define MAX_INSTANCE_ALLOCATIONS 1

namespace FlexFlow {

struct RealmAllocator : public IAllocator {
  RealmAllocator(Legion::Memory);
  ~RealmAllocator() override;

  void *allocate(size_t) override;
  void deallocate(void *) override;

private:
  Legion::Memory memory;
  stack_vector<Realm::RegionInstance, MAX_INSTANCE_ALLOCATIONS> instances;
};

Allocator get_gpu_memory_allocator(Legion::Task const *);

} // namespace FlexFlow

#endif
