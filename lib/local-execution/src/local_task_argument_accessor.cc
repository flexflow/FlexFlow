#include "local_task_argument_accessor.h"

namespace FlexFlow {

ConcreteArgSpec const &
    LocalTaskArgumentAccessor::get_concrete_arg(slot_id name) const {
  return this->slot_argument_map.at(name);
}

PrivilegeTensorAccessor LocalTaskArgumentAccessor::get_tensor(
    slot_id slot, Permissions priv, IsGrad is_grad) const {
  SlotGradId slot_grad_pair = std::make_pair(slot, is_grad);
  auto tensor_backing = std::get<GenericTensorAccessorW>(
      this->slot_tensor_backing_mapping.at(slot_grad_pair));
  if (priv == Permissions::RO) {
    GenericTensorAccessorR readonly_tensor_backing = {
        tensor_backing.data_type, tensor_backing.shape, tensor_backing.ptr};
    return readonly_tensor_backing;
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return tensor_backing;
  } else {
    throw mk_runtime_error("Unhandled privilege mode {}", priv);
  }
}
PrivilegeVariadicTensorAccessor LocalTaskArgumentAccessor::get_variadic_tensor(
    slot_id slot, Permissions priv, IsGrad is_grad) const {
  SlotGradId slot_grad_pair = std::make_pair(slot, is_grad);
  auto variadic_tensor_backing = std::get<std::vector<GenericTensorAccessorW>>(
      this->slot_tensor_backing_mapping.at(slot_grad_pair));
  if (priv == Permissions::RO) {
    std::vector<GenericTensorAccessorR> readonly_variadic_tensor_backing = {};
    for (auto tensor_backing : variadic_tensor_backing) {
      readonly_variadic_tensor_backing.push_back(
          {tensor_backing.data_type, tensor_backing.shape, tensor_backing.ptr});
    }
    return readonly_variadic_tensor_backing;
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return variadic_tensor_backing;
  } else {
    throw mk_runtime_error("Unhandled privilege mode {}", priv);
  }
}

Allocator LocalTaskArgumentAccessor::get_allocator() const {
  return this->allocator;
}

} // namespace FlexFlow
