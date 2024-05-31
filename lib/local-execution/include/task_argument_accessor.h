#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_ARGUMENT_ACCESSOR_H

#include "arg_ref.h"
#include "concrete_arg.h"
#include "config.h"
#include "device_specific.h"
#include "device_states.h"
#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/linear_kernels.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op_task_signature.h"
#include "permissions.h"
#include "tasks.h"
#include "utils/variant.h"
#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

namespace FlexFlow {

template <Permissions>
struct privilege_mode_to_accessor_t {};

template <>
struct privilege_mode_to_accessor_t<Permissions::RW> {
  using type = GenericTensorAccessorW;
};

template <>
struct privilege_mode_to_accessor_t<Permissions::RO> {
  using type = GenericTensorAccessorR;
};

template <>
struct privilege_mode_to_accessor_t<Permissions::WO> {
  using type = GenericTensorAccessorW;
};

template <Permissions PRIV>
using privilege_mode_to_accessor =
    typename privilege_mode_to_accessor_t<PRIV>::type;

using PrivilegeTensorAccessor = std::variant<GenericTensorAccessorR,
                                             GenericTensorAccessorW>;
using PrivilegeVariadicTensorAccessor =
    std::variant<std::vector<GenericTensorAccessorR>,
                 std::vector<GenericTensorAccessorW>>;

struct ITaskArgumentAccessor {
  ITaskArgumentAccessor &operator=(ITaskArgumentAccessor const &) = delete;

  virtual ~ITaskArgumentAccessor() = default;

  virtual ConcreteArgSpec const &get_concrete_arg(slot_id) const = 0;

  virtual PrivilegeTensorAccessor
      get_tensor(slot_id slot, Permissions priv, IsGrad is_grad) const = 0;
  virtual PrivilegeVariadicTensorAccessor get_variadic_tensor(
      slot_id slot, Permissions priv, IsGrad is_grad) const = 0;

  virtual Allocator get_allocator() const = 0;
  virtual size_t get_device_idx() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ITaskArgumentAccessor);

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
    std::function<DeviceStates(TaskArgumentAccessor const &)>,
    std::function<std::optional<float>(TaskArgumentAccessor const &)>>;

template <task_id_t>
TaskImplFunction get_task_impl();

template <task_id_t>
OpTaskSignature get_signature();

} // namespace FlexFlow

#endif
