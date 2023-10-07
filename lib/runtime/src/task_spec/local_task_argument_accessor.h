#ifndef _FLEXFLOW_RUNTIME_SRC_LOCAL_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_RUNTIME_SRC_LOCAL_TASK_ARGUMENT_ACCESSOR_H

#include "accessor.h"
#include "cuda_allocator.h"
#include "sim_environment.h"
#include "device_specific.h"
#include "itask_argument_accessor.h"
#include "realm_allocator.h"
#include "kernels/allocation.h"
#include "pcg/parallel_tensor_guid_t.h"
#include "runtime/config.h"
#include "utils/exception.h"
#include "utils/stack_map.h"
#include "utils/strong_typedef.h"
#include "utils/type_index.h"
#include <vector>

namespace FlexFlow {

struct LocalTaskArgumentAccessor : public ITaskArgumentAccessor {
public:
  template <typename T>
  T const &get_argument(slot_id slot) const;

  PrivilegeType get_tensor(slot_id slot, Permissions priv) const override;

  PrivilegeVariadicType get_variadic_tensor(slot_id slot, Permissions priv) const override;

  template <typename T>
  optional<T> get_optional_argument(slot_id) const;

  template <typename T>
  std::vector<T> get_variadic_argument(slot_id) const;

  PrivilegeType get_tensor_grad(slot_id slot, Permissions priv) const override;

  PrivilegeVariadicType get_variadic_tensor_grad(slot_id slot, Permissions priv) const override;

  size_t get_device_idx() const override;

  LocalTaskArgumentAccessor(SimTaskBinding const &sim_task_binding)
      : sim_task_binding(sim_task_binding), memory_usage(0) {
    local_allocator = Allocator::create<CudaAllocator>();
  }

  size_t get_memory_usage() const {
    return memory_usage;
  }

  void *allocate(size_t size);
  void deallocate(void *ptr);

private:
  SimTaskBinding sim_task_binding;
  Allocator local_allocator;
  size_t memory_usage;
};

} // namespace FlexFlow

#endif