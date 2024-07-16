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

bool are_slots_backings_equivalent_up_to_allocation_addresses(
    TensorSlotsBacking const &slots_1, TensorSlotsBacking const &slots_2) {
  if (slots_1.size() != slots_2.size()) {
    return false;
  }

  auto check_tensors = [](auto acc1_variant, auto acc2_variant) {
    GenericTensorAccessorW acc1 =
        std::get<GenericTensorAccessorW>(acc1_variant);
    GenericTensorAccessorW acc2 =
        std::get<GenericTensorAccessorW>(acc2_variant);
    return is_shape_and_dtype_equal(acc1, acc2);
  };

  auto check_variadic_tensors = [](auto acc1_variant, auto acc2_variant) {
    std::vector<GenericTensorAccessorW> acc1 =
        std::get<std::vector<GenericTensorAccessorW>>(acc1_variant);
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
    return true;
  };

  for (auto const &slot_tensor : slots_1) {
    if (!contains_key(slots_2, slot_tensor.first)) {
      return false;
    }
    auto accessor1_variant = slot_tensor.second;
    auto accessor2_variant = slots_2.at(slot_tensor.first);

    // first check if they hold the same variant type
    if (accessor1_variant.index() != accessor2_variant.index()) {
      return false;
    }
    if (std::holds_alternative<GenericTensorAccessorW>(accessor2_variant)) {
      if (!check_tensors(accessor1_variant, accessor2_variant)) {
        return false;
      }
    } else if (std::holds_alternative<std::vector<GenericTensorAccessorW>>(
                   accessor2_variant)) {
      if (!check_variadic_tensors(accessor1_variant, accessor2_variant)) {
        return false;
      }
    } else {
      throw mk_runtime_error("Unhandled variant type");
    }
  }
  return true;
}

} // namespace FlexFlow
