#include "local-execution/op_task_signature.h"

namespace FlexFlow {

OpTaskSignature::OpTaskSignature(OpTaskType t) : type(t){};

void OpTaskSignature::add_input_slot(int name, SlotType slot_type) {
  this->add_input_slot(slot_id_t{name}, slot_type);
}

void OpTaskSignature::add_input_slot(slot_id_t name, SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = OpTensorSlotSpec{
      name, slot_type, TensorRole::INPUT, IsGrad::NO, OpSlotOptions::NECESSARY};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_optional_input_slot(int name,
                                              SlotType slot_type) {
  this->add_optional_input_slot(slot_id_t{name}, slot_type);
}

void OpTaskSignature::add_optional_input_slot(slot_id_t name,
                                              SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = OpTensorSlotSpec{
      name, slot_type, TensorRole::INPUT, IsGrad::NO, OpSlotOptions::OPTIONAL};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_untrainable_input_slot(int name,
                                                 SlotType slot_type) {
  this->add_untrainable_input_slot(slot_id_t{name}, slot_type);
}

void OpTaskSignature::add_untrainable_input_slot(slot_id_t name,
                                                 SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = OpTensorSlotSpec{name,
                                          slot_type,
                                          TensorRole::INPUT,
                                          IsGrad::NO,
                                          OpSlotOptions::UNTRAINABLE};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_optional_untrainable_input_slot(int name,
                                                          SlotType slot_type) {
  this->add_optional_untrainable_input_slot(slot_id_t{name}, slot_type);
}

void OpTaskSignature::add_optional_untrainable_input_slot(slot_id_t name,
                                                          SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = OpTensorSlotSpec{name,
                                          slot_type,
                                          TensorRole::INPUT,
                                          IsGrad::NO,
                                          OpSlotOptions::OPTIONAL_UNTRAINABLE};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_output_slot(int name, SlotType slot_type) {
  this->add_output_slot(slot_id_t{name}, slot_type);
}

void OpTaskSignature::add_output_slot(slot_id_t name, SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = OpTensorSlotSpec{
      name, slot_type, TensorRole::OUTPUT, IsGrad::NO, OpSlotOptions::OPTIONAL};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_bwd_necessary_output_slot(int name,
                                                    SlotType slot_type) {
  this->add_bwd_necessary_output_slot(slot_id_t{name}, slot_type);
}

void OpTaskSignature::add_bwd_necessary_output_slot(slot_id_t name,
                                                    SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = OpTensorSlotSpec{name,
                                          slot_type,
                                          TensorRole::OUTPUT,
                                          IsGrad::NO,
                                          OpSlotOptions::NECESSARY};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_weight_slot(int name, SlotType slot_type) {
 this->add_weight_slot(slot_id_t{name}, slot_type);
}

void OpTaskSignature::add_weight_slot(slot_id_t name, SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = OpTensorSlotSpec{name,
                                          slot_type,
                                          TensorRole::WEIGHT,
                                          IsGrad::NO,
                                          OpSlotOptions::NECESSARY};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_optional_weight_slot(int name,
                                               SlotType slot_type) {
  this->add_optional_weight_slot(slot_id_t{name}, slot_type);
}

void OpTaskSignature::add_optional_weight_slot(slot_id_t name,
                                               SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = OpTensorSlotSpec{
      name, slot_type, TensorRole::WEIGHT, IsGrad::NO, OpSlotOptions::OPTIONAL};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::set_arg_types(
    std::unordered_map<slot_id_t, std::type_index> const &arg_type) {
  this->task_arg_types = arg_type;
}

void OpTaskSignature::add_from_slot_spec(OpTensorSlotSpec const &spec) {
  this->op_tensor_slots.insert(spec);
}

} // namespace FlexFlow
