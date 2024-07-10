#include "local-execution/op_task_signature.h"

namespace FlexFlow {

OpTaskSignature::OpTaskSignature(OpTaskType t) : type(t){};

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
  OpTensorSlotSpec op_tensor_slot_spec = {
      name, slot_type, TensorRole::OUTPUT, IsGrad::NO, OpSlotOptions::NECESSARY};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_bwd_optional_output_slot(slot_id name,
                                                    SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = {name,
                                          slot_type,
                                          TensorRole::OUTPUT,
                                          IsGrad::NO,
                                          OpSlotOptions::OPTIONAL};
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

void OpTaskSignature::add_grad_slot_from_slot_spec(OpTensorSlotSpec const & spec) {
  OpTensorSlotSpec grad_spec = {
    spec.name, spec.slot_type, spec.tensor_role, IsGrad::YES, spec.slot_option
  };
  this->op_tensor_slots.insert(grad_spec);
}

void OpTaskSignature::infer_from_forward(OpTaskSignature const & fwd) {
  this->return_value = fwd.return_value;
  this->task_arg_types = fwd.task_arg_types;
  this->op_tensor_slots = fwd.op_tensor_slots;
}

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd) {
  OpTaskSignature bwd(OpTaskType::BWD);
  bwd.infer_from_forward(fwd);
  for (auto const & op_tensor_slot_spec : fwd.get_tensor_slots()) {
    OpSlotOptions slot_option = op_tensor_slot_spec.slot_option;
    if (slot_option != OpSlotOptions::UNTRAINABLE || slot_option != OpSlotOptions::OPTIONAL_UNTRAINABLE) {
      bwd.add_grad_slot_from_slot_spec(op_tensor_slot_spec);
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
