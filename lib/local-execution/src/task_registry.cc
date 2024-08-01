#include "local-execution/task_registry.h"

namespace FlexFlow {

void TaskRegistry::register_tasks_for_layer(
    layer_guid_t const &op_id, ComputationGraphOpAttrs const &attrs) {
  this->init_task_ids.insert({op_id, std::nullopt});
  this->forward_task_ids.insert({op_id, std::nullopt});
  this->backward_task_ids.insert({op_id, std::nullopt});

  if (!(attrs.has<WeightAttrs>() || attrs.has<InputAttrs>() ||
        attrs.has<NoopAttrs>())) {
    // register tasks
    std::vector<task_id_t> task_ids = get_task_ids(attrs);
    for (task_id_t task_id : task_ids) {
      TaskSignatureAndImpl task_signature_impl = get_task_sig_impl(task_id);
      switch (task_signature_impl.task_signature.type) {
        case OpTaskType::INIT:
          assert(is_invocation_valid(task_signature_impl.task_signature,
                                     init(attrs)));
          this->init_task_ids[op_id] = task_id;
          break;
        case OpTaskType::FWD:
          assert(is_invocation_valid(task_signature_impl.task_signature,
                                     forward(attrs)));
          this->forward_task_ids[op_id] = task_id;
          break;
        case OpTaskType::BWD:
          assert(is_invocation_valid(task_signature_impl.task_signature,
                                     backward(attrs)));
          this->backward_task_ids[op_id] = task_id;
          break;
        default:
          throw mk_runtime_error("Invalid OpTaskType");
      }
      this->task_mapping.insert({task_id, task_signature_impl});
    }
  }
}

bool TaskRegistry::operator==(TaskRegistry const &other) const {
  return this->tie() == other.tie();
}

bool TaskRegistry::operator!=(TaskRegistry const &other) const {
  return this->tie() != other.tie();
}

std::tuple<std::unordered_map<layer_guid_t, std::optional<task_id_t>> const &,
           std::unordered_map<layer_guid_t, std::optional<task_id_t>> const &,
           std::unordered_map<layer_guid_t, std::optional<task_id_t>> const &,
           std::unordered_map<task_id_t, TaskSignatureAndImpl> const &>
    TaskRegistry::tie() const {
  return std::tie(this->init_task_ids,
                  this->forward_task_ids,
                  this->backward_task_ids,
                  this->task_mapping);
}

std::string format_as(
    std::unordered_map<layer_guid_t, std::optional<task_id_t>> const &x) {
  return fmt::format("TaskRegistry layer_id->task_id");
}

std::ostream &operator<<(
    std::ostream &s,
    std::unordered_map<layer_guid_t, std::optional<task_id_t>> const &x) {
  return (s << fmt::to_string(x));
}

std::string
    format_as(std::unordered_map<task_id_t, TaskSignatureAndImpl> const &x) {
  return fmt::format("TaskRegistry task_id->sig_impl");
}

std::ostream &
    operator<<(std::ostream &s,
               std::unordered_map<task_id_t, TaskSignatureAndImpl> const &x) {
  return (s << fmt::to_string(x));
}

std::string format_as(TaskRegistry const &x) {
  return fmt::format("TaskRegistry");
}

std::ostream &operator<<(std::ostream &s, TaskRegistry const &x) {
  return (s << fmt::to_string(x));
}

} // namespace FlexFlow
