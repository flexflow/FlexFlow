#ifndef _FLEXFLOW_EXECUTION_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_EXECUTION_TASK_ARGUMENT_ACCESSOR_H

#include "slot_id.h"
#include "permissions.h"
#include "kernels/accessor.h"
#include <cstddef>
#include <memory>
#include <optional>
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

using PrivilegeType = std::variant<privilege_mode_to_accessor<Permissions::RW>,
                                   privilege_mode_to_accessor<Permissions::RO>,
                                   privilege_mode_to_accessor<Permissions::WO>>;
using PrivilegeVariadicType =
    std::variant<std::vector<privilege_mode_to_accessor<Permissions::RW>>,
                 std::vector<privilege_mode_to_accessor<Permissions::RO>>,
                 std::vector<privilege_mode_to_accessor<Permissions::WO>>>;

struct ITaskArgumentAccessor {
  ITaskArgumentAccessor &operator=(ITaskArgumentAccessor const &) = delete;

  virtual ~ITaskArgumentAccessor() = 0;

  virtual PrivilegeType get_tensor(slot_id slot, bool is_grad) const = 0;

  virtual PrivilegeVariadicType get_variadic_tensor(slot_id slot,
                                                    Permissions priv) const = 0;

  virtual PrivilegeType get_tensor_grad(slot_id slot,
                                        Permissions priv) const = 0;

  virtual PrivilegeVariadicType
      get_variadic_tensor_grad(slot_id slot, Permissions priv) const = 0;

  virtual size_t get_device_idx() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ITaskArgumentAccessor);

struct TaskArgumentAccessor {
  template <typename T>
  T const &get_argument(slot_id slot) const {
    return this->ptr->get_argument(slot);
  }

  template <typename T>
  std::optional<T> const &get_optional_argument(slot_id slot) const {
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
  TaskArgumentAccessor(std::shared_ptr<ITaskArgumentAccessor const> ptr)
      : ptr(ptr) {}
  std::shared_ptr<ITaskArgumentAccessor const> ptr;
};

} // namespace FlexFlow

#endif
