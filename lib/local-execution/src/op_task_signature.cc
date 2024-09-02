#include "local-execution/op_task_signature.h"
#include "utils/fmt/unordered_map.h"
#include "utils/fmt/unordered_set.h"
#include "utils/fmt/optional.h"

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

void OpTaskSignature::add_optional_input_slot(int name, SlotType slot_type) {
  this->add_optional_input_slot(slot_id_t{name}, slot_type);
}

void OpTaskSignature::add_optional_input_slot(slot_id_t name,
                                              SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = OpTensorSlotSpec{
      name, slot_type, TensorRole::INPUT, IsGrad::NO, OpSlotOptions::OPTIONAL};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_untrainable_input_slot(int name, SlotType slot_type) {
  this->add_untrainable_input_slot(slot_id_t{name}, slot_type);
}

void OpTaskSignature::add_untrainable_input_slot(slot_id_t name,
                                                 SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec =
      OpTensorSlotSpec{name,
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
  OpTensorSlotSpec op_tensor_slot_spec =
      OpTensorSlotSpec{name,
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
  OpTensorSlotSpec op_tensor_slot_spec =
      OpTensorSlotSpec{name,
                       slot_type,
                       TensorRole::OUTPUT,
                       IsGrad::NO,
                       OpSlotOptions::NECESSARY};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_bwd_optional_output_slot(int name,
                                                   SlotType slot_type) {
  this->add_bwd_optional_output_slot(slot_id_t{name}, slot_type);
}

void OpTaskSignature::add_bwd_optional_output_slot(slot_id_t name,
                                                   SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec = OpTensorSlotSpec{
      name, slot_type, TensorRole::OUTPUT, IsGrad::NO, OpSlotOptions::OPTIONAL};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_weight_slot(int name, SlotType slot_type) {
  this->add_weight_slot(slot_id_t{name}, slot_type);
}

void OpTaskSignature::add_weight_slot(slot_id_t name, SlotType slot_type) {
  OpTensorSlotSpec op_tensor_slot_spec =
      OpTensorSlotSpec{name,
                       slot_type,
                       TensorRole::WEIGHT,
                       IsGrad::NO,
                       OpSlotOptions::NECESSARY};
  this->op_tensor_slots.insert(op_tensor_slot_spec);
}

void OpTaskSignature::add_optional_weight_slot(int name, SlotType slot_type) {
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

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd) {
  OpTaskSignature bwd = fwd;
  bwd.type = OpTaskType::BWD;
  for (auto const &op_tensor_slot_spec : fwd.get_tensor_slots()) {
    OpSlotOptions slot_option = op_tensor_slot_spec.slot_option;
    if (slot_option != OpSlotOptions::UNTRAINABLE ||
        slot_option != OpSlotOptions::OPTIONAL_UNTRAINABLE) {
      OpTensorSlotSpec grad_spec =
          OpTensorSlotSpec{op_tensor_slot_spec.name,
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

std::unordered_map<slot_id_t, std::type_index>
    OpTaskSignature::get_arg_types() const {
  return this->task_arg_types;
}

std::string format_as(OpTaskSignature const &x) {
  std::ostringstream oss;
  oss << "<OpTaskSignature";
  oss << " type=" << x.type;
  oss << " return_value=" << x.return_value;
  oss << " task_arg_types=" << fmt::to_string(x.task_arg_types);
  oss << " op_tensor_slots=" << fmt::to_string(x.op_tensor_slots);
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, OpTaskSignature const &x) {
  return s << fmt::to_string(x);
}

} // namespace FlexFlow
