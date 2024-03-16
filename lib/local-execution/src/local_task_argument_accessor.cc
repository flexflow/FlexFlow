#include "local_task_argument_accessor.h"

namespace FlexFlow {

ConcreteArgSpec const & LocalTaskArgumentAccessor::get_concrete_arg(slot_id) const { 
  return this->argument_map[slot_id];
}
OpArgRefTypeBacking const & LocalTaskArgumentAccessor::get_op_arg_ref(slot_id) const { 
  return this->argument_map[slot_id];
  }
RuntimeArgRefTypeBacking const & LocalTaskArgumentAccessor::get_runtime_arg(slot_id) const { 
  return this->argument_map[slot_id]
  }

PrivilegeType LocalTaskArgumentAccessor::get_tensor(slot_id slot, Permissions priv) const { 
  std::pair<slot_id, bool> slot_grad_pair = std::make_pair<slot_id, bool>(slot, false);
  GenericTensorAccessorW tensor_backing = this->tensor_backing_map[slot_grad_pair].visit();
  if (priv == Permissions::RO) {
    GenericTensorAccessorR readonly_tensor_backing = {
        tensor_backing.data_type,
        tensor_backing.get_shape(),
        tensor_backing.ptr};
    return readonly_tensor_backing;
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return tensor_backing;
  } else {
    throw mk_runtime_error("Unhandled privilege mode {}", PRIV);
  }
}
PrivilegeVariadicType LocalTaskArgumentAccessor::get_variadic_tensor(slot_id slot,
                                                  Permissions priv) const { 
  std::pair<slot_id, bool> slot_grad_pair = std::make_pair<slot_id, bool>(slot, false);
  std::vector<GenericTensorAccessorW> variadic_tensor_backing = this->tensor_backing_map[slot_grad_pair].visit();
  if (priv == Permissions::RO) {
    std::vector<GenericTensorAccessorR> readonly_variadic_tensor_backing = {};
    for (auto tensor_backing: variadic_tensor_backing) {
      readonly_variadic_tensor_backing.push_back({
        tensor_backing.data_type,
        tensor_backing.get_shape(),
        tensor_backing.ptr}
      );
    }
    return readonly_variadic_tensor_backing;
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return variadic_tensor_backing;
  } else {
    throw mk_runtime_error("Unhandled privilege mode {}", PRIV);
  }    
}
PrivilegeType LocalTaskArgumentAccessor::get_tensor_grad(slot_id slot,
                                      Permissions priv) const { 
  std::pair<slot_id, bool> slot_grad_pair = std::make_pair<slot_id, bool>(slot, true);
  GenericTensorAccessorW tensor_backing = this->tensor_backing_map[slot_grad_pair].visit();
  if (priv == Permissions::RO) {
    GenericTensorAccessorR readonly_tensor_backing = {
        tensor_backing.data_type,
        tensor_backing.get_shape(),
        tensor_backing.ptr};
    return readonly_tensor_backing;
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return tensor_backing;
  } else {
    throw mk_runtime_error("Unhandled privilege mode {}", PRIV);
  }
}

PrivilegeVariadicType
    LocalTaskArgumentAccessor::get_variadic_tensor_grad(slot_id slot, Permissions priv) const { 
  std::pair<slot_id, bool> slot_grad_pair = std::make_pair<slot_id, bool>(slot, true);
  std::vector<GenericTensorAccessorW> variadic_tensor_backing = this->tensor_backing_map[slot_grad_pair].visit();
  if (priv == Permissions::RO) {
    std::vector<GenericTensorAccessorR> readonly_variadic_tensor_backing = {};
    for (auto tensor_backing: variadic_tensor_backing) {
      readonly_variadic_tensor_backing.push_back({
        tensor_backing.data_type,
        tensor_backing.get_shape(),
        tensor_backing.ptr}
      );
    }
    return readonly_variadic_tensor_backing;
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return variadic_tensor_backing;
  } else {
    throw mk_runtime_error("Unhandled privilege mode {}", PRIV);
  }    
}

Allocator LocalTaskArgumentAccessor::get_allocator() {
  return this->allocator;
}

} // namespace FlexFlow
