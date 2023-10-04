#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_ARGUMENT_ACCESSOR_H

#include "accessor.h"
#include "sim_environment.h"
#include "device_specific.h"
#include "itask_argument_accessor.h"
#include "legion_task_argument_accessor.h"
#include "local_task_argument_accessor.h"
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

using ITaskArgumentAccessorBackend = variant<LegionTaskArgumentAccessor,
                                        LocalTaskArgumentAccessor>;

struct TaskArgumentAccessor {
  template <typename T>
  T const &get_argument(slot_id slot) const {
    return this->ptr->get_argument(slot);
  }

  template <typename T>
  optional<T> const &get_optional_argument(slot_id slot) const {
    return this->ptr->get_optional_argument(slot);
  }

  template <typename T>
  std::vector<T> const &get_variadic_argument(slot_id slot) const {
    return this->ptr->get_variadic_argument(slot);
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor(slot_id slot) const {
    return this->ptr->get_tensor(slot, PRIV);
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_tensor(slot_id slot) const {
    return this->ptr->get_variadic_tensor(slot, PRIV);
  }

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<ITaskArgumentAccessor, T>::value,
                              TaskArgumentAccessor>::type
      create(Args &&...args) {
    return TaskArgumentAccessor(
        std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  TaskArgumentAccessor(std::shared_ptr<ITaskArgumentAccessorBackend const> ptr)
      : ptr(ptr) {}
  std::shared_ptr<ITaskArgumentAccessorBackend const> ptr;
};

} // namespace FlexFlow

#endif
