#include "local-execution/local_task_argument_accessor.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/transform.h"
#include "utils/hash/pair.h"
#include "utils/overload.h"

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
    GenericTensorAccessorR readonly_tensor_backing =
        read_only_accessor_from_write_accessor(tensor_backing);
    return readonly_tensor_backing;
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return tensor_backing;
  } else {
    throw mk_runtime_error(fmt::format("Unhandled privilege mode {}", priv));
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
          read_only_accessor_from_write_accessor(tensor_backing));
    }
    return readonly_variadic_tensor_backing;
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return variadic_tensor_backing;
  } else {
    throw mk_runtime_error(fmt::format("Unhandled privilege mode {}", priv));
  }
}

Allocator LocalTaskArgumentAccessor::get_allocator() const {
  return this->allocator;
}

TensorSlotsBackingWithoutAddresses
    get_slots_backing_without_tensor_allocation_addresses(
        TensorSlotsBacking const &slots_backing) {

  TensorSlotsBackingWithoutAddresses addressless_slots_backing;

  using TensorAccessorVariant =
      std::variant<GenericTensorAccessorW, std::vector<GenericTensorAccessorW>>;
  for (auto const &slot_tensor : slots_backing) {
    TensorAccessorVariant accessor_variant = slot_tensor.second;
    std::visit(
        overload{
            [&](GenericTensorAccessorW const &accessor) {
              addressless_slots_backing.insert(
                  {slot_tensor.first, get_shape_and_datatype(accessor)});
            },
            [&](std::vector<GenericTensorAccessorW> const &variadic_accessor) {
              std::vector<std::pair<ArrayShape, DataType>>
                  variadic_addressless_accessor =
                      transform(variadic_accessor,
                                [](GenericTensorAccessorW const &accessor) {
                                  return get_shape_and_datatype(accessor);
                                });
              addressless_slots_backing.insert(
                  {slot_tensor.first, variadic_addressless_accessor});
            }},
        accessor_variant);
  }
  return addressless_slots_backing;
}

size_t LocalTaskArgumentAccessor::get_device_idx() const {
  return 0;
}

} // namespace FlexFlow
