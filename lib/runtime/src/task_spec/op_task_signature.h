#ifndef _FLEXFLOW_RUNTIME_SRC_OP_TASK_SIGNATURE_H
#define _FLEXFLOW_RUNTIME_SRC_OP_TASK_SIGNATURE_H

#include "task_invocation.h"
#include "task_signature.h"
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
  OpTensorSlotSpec(slot_id, SlotType, TensorRole);

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

  OpTaskType get_task_type() const;

  void add_input_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_optional_input_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_untrainable_input_slot(slot_id,
                                  SlotType slot_type = SlotType::TENSOR);
  void add_optional_untrainable_input_slot(
      slot_id, SlotType slot_type = SlotType::TENSOR);

  void add_output_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_bwd_necessary_output_slot(slot_id,
                                     SlotType slot_type = SlotType::TENSOR);

  void add_weight_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_optional_weight_slot(slot_id, SlotType slot_type = SlotType::TENSOR);

  void add_from_slot_spec(OpTensorSlotSpec const &spec);

  /* void add_input_slot(slot_id, Legion::PrivilegeMode); */
  /* void add_input_slot(slot_id, SlotType, Legion::PrivilegeMode); */

  bool operator==(OpTaskSignature const &) const;
  bool operator!=(OpTaskSignature const &) const;

  template <typename T>
  void add_arg_slot(slot_id name) {
    static_assert(is_serializable<T>::value, "Type must be serializable");
  }

  template <typename T>
  void add_return_value();

  // adds arg_slot without checking is_serializable, used for arguments that are
  // deviceSpecific
  template <typename T>
  void add_unchecked_arg_slot(slot_id name) {
    NOT_IMPLEMENTED();
  }

  std::unordered_set<OpTensorSlotSpec> get_tensor_slots();
  void set_arg_types(std::unordered_map<slot_id, std::type_index> const &);
  std::unordered_map<slot_id, std::type_index> get_arg_types();

private:
  std::unordered_map<slot_id, std::type_index> task_arg_types;
  std::unordered_set<OpTensorSlotSpec> op_tensor_slots;
};

template <task_id_t>
OpTaskSignature get_signature();

template <typename F>
void register_task(task_id_t,
                   std::string const &name,
                   OpTaskSignature const &,
                   F const &func);

template <typename F>
void register_task(task_id_t,
                   std::string const &name,
                   OpTaskSignature const &,
                   F const &func,
                   F const &cpu_func);

} // namespace FlexFlow

#endif
