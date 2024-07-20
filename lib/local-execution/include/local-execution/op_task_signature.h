#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_SIGNATURE_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_SIGNATURE_H

#include "local-execution/is_grad.dtg.h"
#include "local-execution/serialization.h"
#include "local-execution/slot_id.h"
#include "local-execution/slot_type.h"
#include "local-execution/tasks.h"
#include "utils/type_index.h"
#include "utils/visitable.h"

namespace FlexFlow {

enum class TensorRole {
  INPUT,
  WEIGHT,
  OUTPUT,
};

enum class OpTaskType { INIT, FWD, BWD };

enum class OpSlotOptions {
  OPTIONAL,
  UNTRAINABLE,
  OPTIONAL_UNTRAINABLE,
  NECESSARY
};

struct OpTensorSlotSpec {
public:
  OpTensorSlotSpec() = delete;

public:
  slot_id name;
  SlotType slot_type;
  TensorRole tensor_role;
  IsGrad is_grad;
  OpSlotOptions slot_option;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(
    OpTensorSlotSpec, name, slot_type, tensor_role, is_grad, slot_option);

struct OpTaskSignature {
  OpTaskSignature() = delete;
  explicit OpTaskSignature(OpTaskType);
  OpTaskSignature(OpTaskSignature const &other);

  OpTaskType get_task_type() const {
    return this->type;
  }

  void add_input_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_optional_input_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_untrainable_input_slot(slot_id,
                                  SlotType slot_type = SlotType::TENSOR);
  void add_optional_untrainable_input_slot(
      slot_id, SlotType slot_type = SlotType::TENSOR);

  void add_output_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_bwd_optional_output_slot(slot_id,
                                    SlotType slot_type = SlotType::TENSOR);

  void add_weight_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_optional_weight_slot(slot_id, SlotType slot_type = SlotType::TENSOR);

  void add_from_slot_spec(OpTensorSlotSpec const &spec);

  template <typename T>
  void add_arg_slot(slot_id name) {
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
  void add_unchecked_arg_slot(slot_id name) {
    this->task_arg_types.insert({name, get_type_index_for_type<T>()});
  }

  std::unordered_set<OpTensorSlotSpec> get_tensor_slots() const;
  void set_arg_types(std::unordered_map<slot_id, std::type_index> const &);
  std::unordered_map<slot_id, std::type_index> get_arg_types() const;

  OpTaskType type;
  std::optional<std::type_index> return_value;
  std::unordered_map<slot_id, std::type_index> task_arg_types;
  std::unordered_set<OpTensorSlotSpec> op_tensor_slots;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(
    OpTaskSignature, type, return_value, task_arg_types, op_tensor_slots);

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd);

} // namespace FlexFlow

#endif
