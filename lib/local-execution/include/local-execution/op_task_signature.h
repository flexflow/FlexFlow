#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_SIGNATURE_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_SIGNATURE_H

#include "local-execution/is_grad.dtg.h"
#include "local-execution/op_task_type.dtg.h"
#include "local-execution/op_tensor_slot_spec.dtg.h"
#include "local-execution/serialization.h"
#include "local-execution/slot_id_t.dtg.h"
#include "local-execution/slot_type.dtg.h"
#include "local-execution/task_id_t.dtg.h"
#include "utils/type_index.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct OpTaskSignature {
  OpTaskSignature() = delete;
  explicit OpTaskSignature(OpTaskType);

  OpTaskType get_task_type() const {
    return this->type;
  }

  void add_input_slot(int, SlotType slot_type = SlotType::TENSOR);
  void add_input_slot(slot_id_t, SlotType slot_type = SlotType::TENSOR);

  void add_optional_input_slot(int, SlotType slot_type = SlotType::TENSOR);
  void add_optional_input_slot(slot_id_t,
                               SlotType slot_type = SlotType::TENSOR);

  void add_untrainable_input_slot(int, SlotType slot_type = SlotType::TENSOR);
  void add_untrainable_input_slot(slot_id_t,
                                  SlotType slot_type = SlotType::TENSOR);

  void add_optional_untrainable_input_slot(
      int, SlotType slot_type = SlotType::TENSOR);
  void add_optional_untrainable_input_slot(
      slot_id_t, SlotType slot_type = SlotType::TENSOR);

  void add_output_slot(int, SlotType slot_type = SlotType::TENSOR);
  void add_output_slot(slot_id_t, SlotType slot_type = SlotType::TENSOR);

  void add_bwd_optional_output_slot(int, SlotType slot_type = SlotType::TENSOR);
  void add_bwd_optional_output_slot(slot_id_t,
                                    SlotType slot_type = SlotType::TENSOR);

  void add_weight_slot(int, SlotType slot_type = SlotType::TENSOR);
  void add_weight_slot(slot_id_t, SlotType slot_type = SlotType::TENSOR);

  void add_optional_weight_slot(int, SlotType slot_type = SlotType::TENSOR);
  void add_optional_weight_slot(slot_id_t,
                                SlotType slot_type = SlotType::TENSOR);

  void add_from_slot_spec(OpTensorSlotSpec const &spec);

  template <typename T>
  void add_arg_slot(int name) {
    this->add_arg_slot<T>(slot_id_t{name});
  }

  template <typename T>
  void add_arg_slot(slot_id_t name) {
    // static_assert(is_serializable<T>::value, "Type must be serializable");
    this->task_arg_types.insert({name, get_type_index_for_type<T>()});
  }

  template <typename T>
  void add_return_value() {
    this->return_value = get_type_index_for_type<T>();
  }

  // adds arg_slot without checking is_serializable, used for arguments that are
  // deviceSpecific
  template <typename T>
  void add_unchecked_arg_slot(int name) {
    this->add_unchecked_arg_slot<T>(slot_id_t{name});
  }

  // adds arg_slot without checking is_serializable, used for arguments that are
  // deviceSpecific
  template <typename T>
  void add_unchecked_arg_slot(slot_id_t name) {
    this->task_arg_types.insert({name, get_type_index_for_type<T>()});
  }

  std::unordered_set<OpTensorSlotSpec> get_tensor_slots() const;
  void set_arg_types(std::unordered_map<slot_id_t, std::type_index> const &);
  std::unordered_map<slot_id_t, std::type_index> get_arg_types() const;

  OpTaskType type;
  std::optional<std::type_index> return_value;
  std::unordered_map<slot_id_t, std::type_index> task_arg_types;
  std::unordered_set<OpTensorSlotSpec> op_tensor_slots;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(
    OpTaskSignature, type, return_value, task_arg_types, op_tensor_slots);

std::string format_as(OpTaskSignature const &x);
std::ostream &operator<<(std::ostream &s, OpTaskSignature const &x);

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd);

} // namespace FlexFlow

#endif
