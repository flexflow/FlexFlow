#include "local-execution/task_signature.h"

namespace FlexFlow {

TaskSignature make_empty_task_signature() {
  return TaskSignature(std::nullopt, {}, {});
}

void add_slot(TaskSignature &task_signature,
              int name,
              IsGrad is_grad,
              SlotType slot_type) {
  add_slot(task_signature, slot_id_t{name}, is_grad, slot_type);
}

void add_slot(TaskSignature &task_signature,
              slot_id_t name,
              IsGrad is_grad,
              SlotType slot_type) {
  TensorGuidSlotSpec tensor_guid_slot_spec =
      TensorGuidSlotSpec{name, slot_type, is_grad};
  task_signature.tensor_guid_slots.insert(tensor_guid_slot_spec);
}

} // namespace FlexFlow
