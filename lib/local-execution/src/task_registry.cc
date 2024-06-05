#include "local-execution/task_registry.h"

namespace FlexFlow {

void TaskRegistry::register_task(task_id_t const &task_id,
                                 layer_guid_t const &op_id,
                                 ComputationGraphOpAttrs const &attrs) {
  TaskSignatureAndImpl task_signature_impl = get_task_sig_impl(task_id);
  switch (task_signature_impl.task_signature.type) {
    case OpTaskType::INIT:
      assert(
          is_invocation_valid(task_signature_impl.task_signature, init(attrs)));
      this->init_task_ids.insert({op_id, task_id});
      break;
    case OpTaskType::FWD:
      assert(is_invocation_valid(task_signature_impl.task_signature,
                                 forward(attrs)));
      this->forward_task_ids.insert({op_id, task_id});
      break;
    case OpTaskType::BWD:
      assert(is_invocation_valid(task_signature_impl.task_signature,
                                 backward(attrs)));
      this->backward_task_ids.insert({op_id, task_id});
      break;
    default:
      throw mk_runtime_error("Invalid OpTaskType");
  }
  this->task_mapping.insert({task_id, task_signature_impl});
}

} // namespace FlexFlow
