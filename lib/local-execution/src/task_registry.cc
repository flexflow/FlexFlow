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
        throw mk_runtime_error("Invalid OpTaskType, got {}",
                               task_signature_impl.task_signature.type);
    }
    task_registry.task_mapping.insert({task_id, task_signature_impl});
  }
}

bool registry_contains_op_task(TaskRegistry const &task_registry,
                               layer_guid_t const &op,
                               OpTaskType const &op_task_type) {
  std::unordered_map<layer_guid_t, std::optional<task_id_t>> task_ids;
  switch (op_task_type) {
    case OpTaskType::INIT:
      task_ids = task_registry.init_task_ids;
      break;
    case OpTaskType::FWD:
      task_ids = task_registry.forward_task_ids;
      break;
    case OpTaskType::BWD:
      task_ids = task_registry.backward_task_ids;
      break;
    default:
      throw mk_runtime_error("Invalid OpTaskType, got {}", op_task_type);
  }

  return task_ids.at(op).has_value();
}

} // namespace FlexFlow
