#include "local-execution/task_registry.h"
#include "local-execution/task_signature_impl.h"

namespace FlexFlow {

TaskRegistry empty_task_registry() {
  return TaskRegistry{{}, {}, {}, {}};
}

void register_tasks_for_layer(TaskRegistry &task_registry,
                              layer_guid_t const &op_id,
                              ComputationGraphOpAttrs const &attrs) {
  task_registry.init_task_ids.insert({op_id, std::nullopt});
  task_registry.forward_task_ids.insert({op_id, std::nullopt});
  task_registry.backward_task_ids.insert({op_id, std::nullopt});

  // register tasks
  std::vector<task_id_t> task_ids = get_task_ids(attrs);
  for (task_id_t task_id : task_ids) {
    TaskSignatureAndImpl task_signature_impl = get_task_sig_impl(task_id);
    switch (task_signature_impl.task_signature.type) {
      case OpTaskType::INIT:
        assert(is_invocation_valid(task_signature_impl.task_signature,
                                   init(attrs)));
        task_registry.init_task_ids[op_id] = task_id;
        break;
      case OpTaskType::FWD:
        assert(is_invocation_valid(task_signature_impl.task_signature,
                                   forward(attrs)));
        task_registry.forward_task_ids[op_id] = task_id;
        break;
      case OpTaskType::BWD:
        assert(is_invocation_valid(task_signature_impl.task_signature,
                                   backward(attrs)));
        task_registry.backward_task_ids[op_id] = task_id;
        break;
      default:
        throw mk_runtime_error("Invalid OpTaskType");
    }
    task_registry.task_mapping.insert({task_id, task_signature_impl});
  }
}

} // namespace FlexFlow
