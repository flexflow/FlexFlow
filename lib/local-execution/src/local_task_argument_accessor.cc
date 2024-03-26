#include "local_task_argument_accessor.h"

namespace FlexFlow {

ConcreteArgSpec const & LocalTaskArgumentAccessor::get_concrete_arg(slot_id name) const { 
  return std::get<ConcreteArgSpec>(this->argument_map.at(name));
}
OpArgRefTypeBacking const & LocalTaskArgumentAccessor::get_op_arg_ref(slot_id name) const { 
  return std::get<OpArgRefTypeBacking>(this->argument_map.at(name));
}
RuntimeArgRefTypeBacking const & LocalTaskArgumentAccessor::get_runtime_arg(slot_id name) const { 
  return std::get<RuntimeArgRefTypeBacking>(this->argument_map.at(name));
}

PrivilegeType LocalTaskArgumentAccessor::get_tensor(slot_id slot, Permissions priv, IsGrad is_grad) const { 
  std::pair<slot_id, IsGrad> slot_grad_pair = std::make_pair(slot, is_grad);
  GenericTensorAccessorW tensor_backing = std::get<GenericTensorAccessorW>(this->tensor_backing_map.at(slot_grad_pair));
  if (priv == Permissions::RO) {
    GenericTensorAccessorR readonly_tensor_backing = {
        tensor_backing.data_type,
        tensor_backing.shape,
        tensor_backing.ptr};
    return readonly_tensor_backing;
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return tensor_backing;
  } else {
    throw mk_runtime_error("Unhandled privilege mode {}", priv);
  }
}
PrivilegeVariadicType LocalTaskArgumentAccessor::get_variadic_tensor(slot_id slot,
                                                  Permissions priv, IsGrad is_grad) const { 
  std::pair<slot_id, IsGrad> slot_grad_pair = std::make_pair(slot, is_grad);
  std::vector<GenericTensorAccessorW> variadic_tensor_backing = std::get<std::vector<GenericTensorAccessorW>>(this->tensor_backing_map.at(slot_grad_pair));
  if (priv == Permissions::RO) {
    std::vector<GenericTensorAccessorR> readonly_variadic_tensor_backing = {};
    for (auto tensor_backing: variadic_tensor_backing) {
      readonly_variadic_tensor_backing.push_back({
        tensor_backing.data_type,
        tensor_backing.shape,
        tensor_backing.ptr}
      );
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
