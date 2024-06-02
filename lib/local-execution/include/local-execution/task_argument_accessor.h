#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_ARGUMENT_ACCESSOR_H

#include "itask_argument_accessor.h"
#include "local-execution/device_specific.h"
#include "local-execution/device_states.h"
#include "local-execution/tasks.h"
#include "utils/variant.h"

namespace FlexFlow {

struct TaskArgumentAccessor {
  template <typename T>
  T const &get_argument(slot_id slot) const {
    return this->ptr->get_concrete_arg(slot).get<T>();
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor(slot_id slot) const {
    return std::get<privilege_mode_to_accessor<PRIV>>(
        this->ptr->get_tensor(slot, PRIV, IsGrad::NO));
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor_grad(slot_id slot) const {
    return std::get<privilege_mode_to_accessor<PRIV>>(
        this->ptr->get_tensor(slot, PRIV, IsGrad::YES));
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_tensor(slot_id slot) const {
    return std::get<std::vector<privilege_mode_to_accessor<PRIV>>>(
        this->ptr->get_variadic_tensor(slot, PRIV, IsGrad::NO));
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_tensor_grad(slot_id slot) const {
    return std::get<std::vector<privilege_mode_to_accessor<PRIV>>>(
        this->ptr->get_variadic_tensor(slot, PRIV, IsGrad::YES));
  }

  Allocator get_allocator() const {
    return this->ptr->get_allocator();
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
  TaskArgumentAccessor(std::shared_ptr<ITaskArgumentAccessor const> ptr)
      : ptr(ptr) {}
  std::shared_ptr<ITaskArgumentAccessor const> ptr;
};

using TaskImplFunction = std::variant<
    std::function<DeviceSpecific<DeviceStates>(TaskArgumentAccessor const &)>,
    std::function<std::optional<float>(TaskArgumentAccessor const &)>>;

template <task_id_t>
TaskImplFunction get_task_impl();

template <task_id_t>
OpTaskSignature get_signature();

} // namespace FlexFlow

#endif
