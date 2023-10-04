#include "op_task_signature.h"

namespace FlexFlow {

OpTaskSignature::OpTaskSignature(OpTaskType t) {
  this->type = t;
}

void OpTaskSignature::add_input_slot(slot_id name, SlotType slot_type = SlotType::TENSOR) {
  OpTensorSlotSpec op_tensor_slot_spec {
    name, slot_type, TensorRole::INPUT, IsGrad::NO, OpSlotOptions::NECESSARY
  };
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_optional_input_slot(slot_id name, SlotType slot_type = SlotType::TENSOR) {
  OpTensorSlotSpec op_tensor_slot_spec {
    name, slot_type, TensorRole::INPUT, IsGrad::NO, OpSlotOptions::OPTIONAL
  };
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_untrainable_input_slot(slot_id name, SlotType slot_type = SlotType::TENSOR) {
  OpTensorSlotSpec op_tensor_slot_spec {
    name, slot_type, TensorRole::INPUT, IsGrad::NO, OpSlotOptions::UNTRAINABLE
  };
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_optional_untrainable_input_slot(slot_id name, SlotType slot_type = SlotType::TENSOR) {
  OpTensorSlotSpec op_tensor_slot_spec {
    name, slot_type, TensorRole::INPUT, IsGrad::NO, OpSlotOptions::OPTIONAL_UNTRAINABLE
  };
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_output_slot(slot_id name, SlotType slot_type = SlotType::TENSOR) {
  OpTensorSlotSpec op_tensor_slot_spec {
    name, slot_type, TensorRole::OUTPUT, IsGrad::NO, OpSlotOptions::OPTIONAL
  };
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_bwd_necessary_output_slot(slot_id name, SlotType slot_type = SlotType::TENSOR) {
  OpTensorSlotSpec op_tensor_slot_spec {
    name, slot_type, TensorRole::OUTPUT, IsGrad::NO, OpSlotOptions::NECESSARY
  };
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_weight_slot(slot_id name, SlotType slot_type = SlotType::TENSOR) {
  OpTensorSlotSpec op_tensor_slot_spec {
    name, slot_type, TensorRole::WEIGHT, IsGrad::NO, OpSlotOptions::NECESSARY
  };
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_optional_weight_slot(slot_id name, SlotType slot_type = SlotType::TENSOR) {
  OpTensorSlotSpec op_tensor_slot_spec {
    name, slot_type, TensorRole::WEIGHT, IsGrad::NO, OpSlotOptions::OPTIONAL
  };
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::set_arg_types(std::unordered_map<slot_id, std::type_index> const & arg_type) {
  this->task_arg_types = arg_type;
}

void OpTaskSignature::add_from_slot_spec(OpTensorSlotSpec const &spec) {
  this->op_tensor_slots.insert(spec);
}

OpTaskSignature get_op_signature(task_id_t const &task_id) {
  return OpTaskSignature::task_sig_map.at(task_id);
}

}