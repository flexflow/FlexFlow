#include "local-execution/op_task_signature.h"

namespace FlexFlow {

OpTaskSignature::OpTaskSignature(OpTaskType t) : type(t){};

OpTaskSignature::OpTaskSignature(OpTaskSignature const &other)
    : type(other.type), return_value(other.return_value),
      task_arg_types(task_arg_types), op_tensor_slots(op_tensor_slots){};

void OpTaskSignature::add_input_slot(slot_id name, SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = {
      name, slot_type, TensorRole::INPUT, IsGrad::NO, OpSlotOptions::NECESSARY};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_optional_input_slot(slot_id name,
                                              SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = {
      name, slot_type, TensorRole::INPUT, IsGrad::NO, OpSlotOptions::OPTIONAL};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_untrainable_input_slot(slot_id name,
                                                 SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = {name,
                                          slot_type,
                                          TensorRole::INPUT,
                                          IsGrad::NO,
                                          OpSlotOptions::UNTRAINABLE};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_optional_untrainable_input_slot(slot_id name,
                                                          SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = {name,
                                          slot_type,
                                          TensorRole::INPUT,
                                          IsGrad::NO,
                                          OpSlotOptions::OPTIONAL_UNTRAINABLE};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_output_slot(slot_id name, SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = {name,
                                          slot_type,
                                          TensorRole::OUTPUT,
                                          IsGrad::NO,
                                          OpSlotOptions::NECESSARY};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_bwd_optional_output_slot(slot_id name,
                                                   SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = {
      name, slot_type, TensorRole::OUTPUT, IsGrad::NO, OpSlotOptions::OPTIONAL};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_weight_slot(slot_id name, SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = {name,
                                          slot_type,
                                          TensorRole::WEIGHT,
                                          IsGrad::NO,
                                          OpSlotOptions::NECESSARY};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_optional_weight_slot(slot_id name,
                                               SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = {
      name, slot_type, TensorRole::WEIGHT, IsGrad::NO, OpSlotOptions::OPTIONAL};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::set_arg_types(
    std::unordered_map<slot_id, std::type_index> const &arg_type) {
  this->task_arg_types = arg_type;
}

void OpTaskSignature::add_from_slot_spec(OpTensorSlotSpec const &spec) {
  this->op_tensor_slots.insert(spec);
}

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd) {
  OpTaskSignature bwd(fwd);
  for (auto const &op_tensor_slot_spec : fwd.get_tensor_slots()) {
    OpSlotOptions slot_option = op_tensor_slot_spec.slot_option;
    if (slot_option != OpSlotOptions::UNTRAINABLE ||
        slot_option != OpSlotOptions::OPTIONAL_UNTRAINABLE) {
      OpTensorSlotSpec grad_spec = {op_tensor_slot_spec.name,
                                    op_tensor_slot_spec.slot_type,
                                    op_tensor_slot_spec.tensor_role,
                                    IsGrad::YES,
                                    op_tensor_slot_spec.slot_option};
      bwd.op_tensor_slots.insert(grad_spec);
    }
  }

  return bwd;
}

std::unordered_set<OpTensorSlotSpec> OpTaskSignature::get_tensor_slots() const {
  return this->op_tensor_slots;
}

std::unordered_map<slot_id, std::type_index>
    OpTaskSignature::get_arg_types() const {
  return this->task_arg_types;
}

} // namespace FlexFlow
