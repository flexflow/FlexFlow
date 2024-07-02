#include "local-execution/local_task_argument_accessor.h"

namespace FlexFlow {

ConcreteArgSpec const &
    LocalTaskArgumentAccessor::get_concrete_arg(slot_id name) const {
  return this->arg_slots_backing.at(name);
}

GenericTensorAccessor LocalTaskArgumentAccessor::get_tensor(
    slot_id slot, Permissions priv, IsGrad is_grad) const {
  SlotGradId slot_grad_pair = std::make_pair(slot, is_grad);
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
    slot_id slot, Permissions priv, IsGrad is_grad) const {
  SlotGradId slot_grad_pair = std::make_pair(slot, is_grad);
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

bool are_slots_backings_virtually_equivalent(
    TensorSlotsBacking const &slots_1, TensorSlotsBacking const &slots_2) {
  if (slots_1.size() != slots_2.size()) {
    return false;
  }

  for (auto const &pairing : slots_1) {
    if (!contains_key(slots_2, pairing.first)) {
      return false;
    }
    auto acc2_variant = slots_2.at(pairing.first);
    if (acc2_variant.index() != pairing.second.index()) {
      return false;
    }
    if (std::holds_alternative<GenericTensorAccessorW>(acc2_variant)) {
      GenericTensorAccessorW acc1 =
          std::get<GenericTensorAccessorW>(pairing.second);
      GenericTensorAccessorW acc2 =
          std::get<GenericTensorAccessorW>(acc2_variant);
      if (!is_shape_and_dtype_equal(acc1, acc2)) {
        return false;
      }
    } else {
      std::vector<GenericTensorAccessorW> acc1 =
          std::get<std::vector<GenericTensorAccessorW>>(pairing.second);
      std::vector<GenericTensorAccessorW> acc2 =
          std::get<std::vector<GenericTensorAccessorW>>(acc2_variant);
      if (acc1.size() != acc2.size()) {
        return false;
      }
      for (int i = 0; i < acc1.size(); ++i) {
        if (!is_shape_and_dtype_equal(acc1.at(i), acc2.at(i))) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace FlexFlow
