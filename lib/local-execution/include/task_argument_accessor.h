#ifndef _FLEXFLOW_EXECUTION_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_EXECUTION_TASK_ARGUMENT_ACCESSOR_H

#include "arg_ref.h"
#include "concrete_arg.h"
#include "config.h"
#include "device_specific.h"
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

using PrivilegeType =
    std::variant<GenericTensorAccessorR, GenericTensorAccessorW>;
using PrivilegeVariadicType = std::variant<std::vector<GenericTensorAccessorR>,
                                           std::vector<GenericTensorAccessorW>>;

// TODO: define device state variant in another file
using DeviceStates = std::variant<LinearPerDeviceState>;

using OpArgRefTypeBacking =
    std::variant<ParallelTensorShape, DeviceSpecific<DeviceStates>>;
using RuntimeArgRefTypeBacking = std::variant<ProfilingSettings,
                                              DeviceSpecific<PerDeviceFFHandle>,
                                              FFIterationConfig>;

using ArgRefBacking = std::
    variant<OpArgRefTypeBacking, RuntimeArgRefTypeBacking, ConcreteArgSpec>;

struct ITaskArgumentAccessor {
  ITaskArgumentAccessor &operator=(ITaskArgumentAccessor const &) = delete;

  virtual ~ITaskArgumentAccessor() = default;

  virtual ConcreteArgSpec const &get_concrete_arg(slot_id) const = 0;
  virtual OpArgRefTypeBacking const &get_op_arg_ref(slot_id) const = 0;
  virtual RuntimeArgRefTypeBacking const &get_runtime_arg(slot_id) const = 0;

  virtual PrivilegeType
      get_tensor(slot_id slot, Permissions priv, IsGrad is_grad) const = 0;
  virtual PrivilegeVariadicType get_variadic_tensor(slot_id slot,
                                                    Permissions priv,
                                                    IsGrad is_grad) const = 0;

  virtual Allocator get_allocator() const = 0;
  virtual size_t get_device_idx() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ITaskArgumentAccessor);

struct TaskArgumentAccessor {
  template <typename T>
  T const &get_argument(slot_id slot) const {
    if constexpr (is_in_variant<T, OpArgRefTypeBacking>::value) {
      return std::get<T>(this->ptr->get_op_arg_ref(slot));
    } else if constexpr (is_in_variant<T, RuntimeArgRefTypeBacking>::value) {
      return std::get<T>(this->ptr->get_runtime_arg(slot));
    } else {
      return this->ptr->get_concrete_arg(slot).get<T>();
    }
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

using DeviceStates = std::variant<LinearPerDeviceState>;

using TaskImplFunction = std::variant<
    std::function<DeviceStates(TaskArgumentAccessor const &)>,
    std::function<std::optional<float>(TaskArgumentAccessor const &)>>;

template <task_id_t>
TaskImplFunction get_task_impl();

template <task_id_t>
OpTaskSignature get_signature();

} // namespace FlexFlow

#endif
