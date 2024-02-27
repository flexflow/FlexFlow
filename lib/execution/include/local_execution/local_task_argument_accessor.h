#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_ARGUMENT_ACCESSOR_H

#include "accessor.h"
#include "kernels/allocation.h"
#include "task_argument_accessor.h"
#include <unordered_map>

namespace FlexFlow {

using SlotGradId = std::pair<slot_id, IsGrad>;

struct LocalTaskArgumentAccessor : public ITaskArgumentAccessor {

  LocalTaskArgumentAccessor(Allocator);

  template <typename T>
  T const &get_argument(slot_id) const;

  template <typename T>
  optional<T> get_optional_argument(slot_id) const;

  template <typename T>
  std::vector<T> get_variadic_argument(slot_id) const;

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor(slot_id slot, bool is_grad) const;

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor_grad(slot_id slot) const;

  Allocator get_allocator();

  void insert_tensor(SlotGradId tensor_id,
                     GenericTensorAccessorW tensor_backing) {
    this->tensor_backing_map.insert({tensor_id, tensor_backing});
  }

private:
  Allocator allocator;
  std::unordered_map<SlotGradId, GenericTensorAccessorW> tensor_backing_map;

  template <typename T>
  std::unordered_map<slot_id, T> argument_map;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalTaskArgumentAccessor);

} // namespace FlexFlow

#endif
