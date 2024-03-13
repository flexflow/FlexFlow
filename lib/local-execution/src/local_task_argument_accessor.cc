#include "local_task_argument_accessor.h"

namespace FlexFlow {

  ConcreteArgSpec const & LocalTaskArgumentAccessor::get_concrete_arg(slot_id) const { NOT_IMPLEMENTED(); }
  OpArgRefTypeBacking const & LocalTaskArgumentAccessor::get_op_arg_ref(slot_id) const { NOT_IMPLEMENTED(); }
  RuntimeArgRefTypeBacking const & LocalTaskArgumentAccessor::get_runtime_arg(slot_id) const { NOT_IMPLEMENTED(); }

  PrivilegeType LocalTaskArgumentAccessor::get_tensor(slot_id slot, Permissions priv) const { NOT_IMPLEMENTED(); }
  PrivilegeVariadicType LocalTaskArgumentAccessor::get_variadic_tensor(slot_id slot,
                                                    Permissions priv) const { NOT_IMPLEMENTED(); }
  PrivilegeType LocalTaskArgumentAccessor::get_tensor_grad(slot_id slot,
                                        Permissions priv) const { NOT_IMPLEMENTED(); }
  PrivilegeVariadicType
      LocalTaskArgumentAccessor::get_variadic_tensor_grad(slot_id slot, Permissions priv) const { NOT_IMPLEMENTED(); }

// template <Permissions PRIV>
// privilege_mode_to_accessor<PRIV>
//     LocalTaskArgumentAccessor::get_tensor(slot_id slot,
//                                           bool is_grad = false) const {
//   std::pair<slot_id, bool> slot_grad_pair =
//       std::make_pair<slot_id, bool>(slot, is_grad);
//   GenericTensorAccessorW tensor_backing =
//       this->tensor_backing_map[slot_grad_pair];
//   if (PRIV == Permissions::RO) {
//     GenericTensorAccessorR readonly_tensor_backing = {
//         tensor_backing.data_type,
//         tensor_backing.get_shape(),
//         tensor_backing.ptr};
//     return readonly_tensor_backing;
//   } else if (PRIV == Permissions::RW || PRIV == Permissions::WO) {
//     return tensor_backing;
//   } else {
//     throw mk_runtime_error("Unhandled privilege mode {}", PRIV);
//   }
// }

// template <Permissions PRIV>
// privilege_mode_to_accessor<PRIV>
//     LocalTaskArgumentAccessor::get_tensor_grad(slot_id slot) const {
//   return this->get_tensor(slot_id, true);
// }

Allocator LocalTaskArgumentAccessor::get_allocator() {
  return this->allocator;
}

} // namespace FlexFlow
