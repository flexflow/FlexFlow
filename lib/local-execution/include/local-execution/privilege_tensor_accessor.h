#ifndef _FLEXFLOW_LOCAL_EXECUTION_PRIVILEGE_TENSOR_ACCESSOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_PRIVILEGE_TENSOR_ACCESSOR_H

#include "kernels/accessor.h"
#include "local-execution/permissions.h"

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

using PrivilegeTensorAccessor =
    std::variant<GenericTensorAccessorR, GenericTensorAccessorW>;
using PrivilegeVariadicTensorAccessor =
    std::variant<std::vector<GenericTensorAccessorR>,
                 std::vector<GenericTensorAccessorW>>;

} // namespace FlexFlow

#endif
