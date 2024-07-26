#include "local-execution/local_task_argument_accessor.h"
#include "utils/hash/pair.h"

namespace FlexFlow {

LocalTaskArgumentAccessor::LocalTaskArgumentAccessor(
    Allocator const &allocator,
    TensorSlotsBacking const &tensor_slots_backing,
    ArgSlotsBacking const &arg_slots_backing)
    : allocator(allocator), tensor_slots_backing(tensor_slots_backing),
      arg_slots_backing(arg_slots_backing){};

ConcreteArgSpec const &
    LocalTaskArgumentAccessor::get_concrete_arg(slot_id_t name) const {
  return this->arg_slots_backing.at(name);
}

GenericTensorAccessor LocalTaskArgumentAccessor::get_tensor(
    slot_id_t slot, Permissions priv, IsGrad is_grad) const {
  SlotGradId slot_grad_pair = SlotGradId{slot, is_grad};
  auto tensor_backing = std::get<GenericTensorAccessorW>(
      this->tensor_slots_backing.at(slot_grad_pair));
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
VariadicGenericTensorAccessor LocalTaskArgumentAccessor::get_variadic_tensor(
    slot_id_t slot, Permissions priv, IsGrad is_grad) const {
  SlotGradId slot_grad_pair = SlotGradId{slot, is_grad};
  auto variadic_tensor_backing = std::get<std::vector<GenericTensorAccessorW>>(
      this->tensor_slots_backing.at(slot_grad_pair));
  if (priv == Permissions::RO) {
    std::vector<GenericTensorAccessorR> readonly_variadic_tensor_backing = {};
    for (GenericTensorAccessorW const &tensor_backing :
         variadic_tensor_backing) {
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

size_t LocalTaskArgumentAccessor::get_device_idx() const {
  return 0;
}

} // namespace FlexFlow
